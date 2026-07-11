import os
import sys
import pickle
import gzip
import json
from pathlib import Path
from datetime import datetime
import smtplib
from email.message import EmailMessage
import concurrent.futures
import gc

import numpy as np

# Import Plotting Modules
from plots.pos_fft import plot_node_pos_moment_fft
from plots.rotation import plot_angular_velocity, plot_axis_angle_over_time
from plots.energy import plot_energies

# Helper Functions (Energy)
def E_T(node, mass):
    velocity = np.array(node[1])
    return 0.5 * mass * np.dot(velocity, velocity)

def E_R(omega_vec, inertia):
    w = np.array(omega_vec)
    I = np.array(inertia)
    if I.ndim == 1: 
        return 0.5 * np.dot(w, I * w)
    return 0.5 * np.dot(w, I @ w)

def E_PB(kappa, K_bt):
    k = np.array(kappa)
    K = np.array(K_bt)
    if K.ndim == 1:
        return 0.5 * np.dot(k, K * k)
    return 0.5 * np.dot(k, K @ k)

def E_PS(strain, K_se):
    s = np.array(strain)
    K = np.array(K_se)
    if K.ndim == 1:
        return 0.5 * np.dot(s, K * s)
    return 0.5 * np.dot(s, K @ s)

def dynamic_scaling(edge):
    len_inv = edge[1][3]      # edges[current].len_inv
    e = 1.0 / (len_inv * dl)
    eta = 1.0 - 0.3 * (e - 1.0)
    return eta

# Parallel Extraction Worker
def _process_single_chunk(args):
    nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt, dl = args
    
    node_data = {n: {'pos': [], 'vel': [], 'mom': [], 'quats': [], 'omega': []} for n in inspect_nodes}
    t_ke, r_ke, bt_pe, ss_pe = [], [], [], []

    with gzip.open(nf, "rb") as f:
        n_chunk = pickle.load(f)
    with gzip.open(ef, "rb") as f:
        e_chunk = pickle.load(f)

    #reconstruct reference vector
    node_count = len(n_chunk[0])
    tuned_length = dl * (node_count - 1)
    x = np.linspace(0.0, tuned_length, node_count, dtype=np.float32)
    reference_positions = np.stack(
        [x, np.zeros_like(x), np.zeros_like(x)],
        axis=1,
    ).astype(np.float32)

    reference_vectors = np.array(
        [reference_positions[i + 1] - reference_positions[i]
         for i in range(node_count - 1)],
        dtype=np.float32,
    )

    for i in range(len(n_chunk)):
        n_frame, e_frame = n_chunk[i], e_chunk[i]
        
        # Extract specific nodes
        for n in inspect_nodes:
            target_node = n_frame[n]
            node_data[n]['pos'].append(target_node[0])
            node_data[n]['vel'].append(target_node[1])
            node_data[n]['mom'].append(target_node[2])
            
            target_edge_idx = min(n, len(e_frame)-1)
            node_data[n]['quats'].append(e_frame[target_edge_idx][0])
            node_data[n]['omega'].append(e_frame[target_edge_idx][1][0])
        
        edge_stretch = np.empty(len(e_frame), dtype=np.float32)
        for k in range(len(e_frame)-1): 
            r = (
                reference_vectors[k]
                + np.asarray(n_frame[k + 1][0], dtype=np.float32)
                - np.asarray(n_frame[k][0], dtype=np.float32)
            )

            len_inv = np.float32(1.0) / np.linalg.norm(r)
            stretch = np.float32(1.0) / (len_inv * np.float32(dl))

            edge_stretch[k] = stretch

        t_sum = 0.0
        bt_sum = 0.0
        for j, node in enumerate(n_frame):
            weight = 0.5 if (j == 0 or j == node_count - 1) else 1.0
            t_sum += E_T(node, m_node) * weight
            if 0 < j < node_count - 1:
                epsilon = np.float32(0.5) * (
                    edge_stretch[j - 1] + edge_stretch[j]
                )
                eta = np.float32(1.0) - np.float32(0.3) * (
                    epsilon - np.float32(1.0)
                )
                K_bt_dyn = K_bt * (eta ** 4)
                bt_sum += E_PB(node[3], K_bt_dyn)

        r_sum = 0.0
        ss_sum = 0.0
        for k, edge in enumerate(e_frame[:-1]): # Renamed 'i' to 'k' here too
            _, edge_vecs = edge
            eta = np.float32(1.0) - np.float32(0.3) * (
                edge_stretch[k] - np.float32(1.0)
            )
            inertia_dyn = inertia * (eta ** 2)
            K_se_dyn = K_se * (eta ** 2)
            r_sum += E_R(edge_vecs[0], inertia_dyn)
            ss_sum += E_PS(edge_vecs[2], K_se_dyn)
               
        t_ke.append(t_sum)
        bt_pe.append(bt_sum * dl)
        r_ke.append(r_sum)
        ss_pe.append(ss_sum * dl)

    # Convert to numpy arrays directly to minimize IPC overhead
    out_node_data = {}
    for n in inspect_nodes:
        out_node_data[n] = {
            'pos': np.array(node_data[n]['pos'], dtype=np.float32),
            'vel': np.array(node_data[n]['vel'], dtype=np.float32),
            'mom': np.array(node_data[n]['mom'], dtype=np.float32),
            'quats': np.array(node_data[n]['quats'], dtype=np.float32),
            'omega': np.array(node_data[n]['omega'], dtype=np.float32)
        }

    return (len(n_chunk), out_node_data,
            np.array(t_ke, dtype=np.float32), np.array(r_ke, dtype=np.float32), 
            np.array(bt_pe, dtype=np.float32), np.array(ss_pe, dtype=np.float32))

# Core Memory-Efficient Extraction
def process_and_plot(sim_path, run_id, config, email_config, m_node, dl, inertia, K_se, K_bt, inspect_nodes):
    sim_path = Path(sim_path)
    node_files = sorted(sim_path.glob("n_*.pkl.gz"))
    edge_files = sorted(sim_path.glob("e_*.pkl.gz"))
    
    dt = config.get("dt", 1e-5)
    oversamp = config.get("oversampling_factor", 100)
    
    print(f"[{datetime.now()}] Dispatching data extraction for {len(node_files)} chunks across workers...")
    print(f"[{datetime.now()}] Target nodes for inspection: {inspect_nodes}")
    
    # 1. Fully Parallelized Extraction
    args_list = [(nf, ef, inspect_nodes, m_node, inertia, K_se, K_bt, dl) for nf, ef in zip(node_files, edge_files)]
    
    all_node_data = {n: {'pos': [], 'vel': [], 'mom': [], 'quats': [], 'omega': []} for n in inspect_nodes}
    all_tke, all_rke, all_bpe, all_spe = [], [], [], []
    
    excitation_cutoff_frame = 0
    frames_processed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        for file_idx, res in enumerate(executor.map(_process_single_chunk, args_list)):
            n_frames, nd_dict, c_tke, c_rke, c_bpe, c_spe = res
            
            for n in inspect_nodes:
                all_node_data[n]['pos'].append(nd_dict[n]['pos'])
                all_node_data[n]['vel'].append(nd_dict[n]['vel'])
                all_node_data[n]['mom'].append(nd_dict[n]['mom'])
                all_node_data[n]['quats'].append(nd_dict[n]['quats'])
                all_node_data[n]['omega'].append(nd_dict[n]['omega'])
                
            all_tke.append(c_tke)
            all_rke.append(c_rke)
            all_bpe.append(c_bpe)
            all_spe.append(c_spe)
            
            frames_processed += n_frames
            if file_idx <= 9: # Files 0 through 9 represent the first 10 dispatches
                excitation_cutoff_frame += n_frames

    print(f"[{datetime.now()}] Extraction complete. Stitching arrays...")
    time = np.arange(frames_processed) * (dt * oversamp)
    
    stitched_node_data = {}
    for n in inspect_nodes:
        stitched_node_data[n] = {
            'pos': np.concatenate(all_node_data[n]['pos']),
            'vel': np.concatenate(all_node_data[n]['vel']),
            'mom': np.concatenate(all_node_data[n]['mom']),
            'quats': np.concatenate(all_node_data[n]['quats']),
            'omega': np.concatenate(all_node_data[n]['omega'])
        }
        
    t_ke = np.concatenate(all_tke)
    r_ke = np.concatenate(all_rke)
    bt_pe = np.concatenate(all_bpe)
    ss_pe = np.concatenate(all_spe)

    # Free up collection lists aggressively
    del all_node_data, all_tke, all_rke, all_bpe, all_spe
    gc.collect()

    # Prepare Plotting Tasks for Multiprocessing
    output_dir = sim_path / "plots"
    output_dir.mkdir(exist_ok=True)
    
    phases = {
        "Excitation": (0, excitation_cutoff_frame),
        "Free_Vibration": (excitation_cutoff_frame, None),
        "Full_Simulation": (0, None)
    }

    plot_tasks = []

    for phase_name, (start, end) in phases.items():
        p_time = time[start:end]
        p_tke, p_rke, p_bpe, p_spe = t_ke[start:end], r_ke[start:end], bt_pe[start:end], ss_pe[start:end]
        
        energy_prefix = str(output_dir / f"run_{run_id:03d}_{phase_name}")
        phase_title = phase_name.replace("_", " ")

        # Queue Global Energy Plots (Only once per phase)
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe), 
                           {"print_totals": (phase_name == "Full_Simulation"), "title_prefix": phase_title, "filename": f"{energy_prefix}_energy.png"}))
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe), 
                           {"normalized": True, "title_prefix": phase_title, "filename": f"{energy_prefix}_energy_norm.png"}))
        plot_tasks.append((plot_energies, (p_time, p_tke, p_rke, p_bpe, p_spe), 
                           {"mode_transfer": True, "title_prefix": phase_title, "filename": f"{energy_prefix}_energy_mode.png"}))

        # Queue Node-Specific Plots
        for n in inspect_nodes:
            p_pos = stitched_node_data[n]['pos'][start:end]
            p_mom = stitched_node_data[n]['mom'][start:end]
            p_quats = stitched_node_data[n]['quats'][start:end]
            p_omega = stitched_node_data[n]['omega'][start:end]
            
            node_prefix = str(output_dir / f"run_{run_id:03d}_node_{n:03d}_{phase_name}")
            node_title = f"Node {n} | {phase_title}"

            plot_tasks.append((plot_node_pos_moment_fft, (p_time, p_pos, p_mom, dt, oversamp), 
                               {"moments": True, "title_prefix": node_title, "filename_prefix": f"{node_prefix}_fft"}))

            plot_tasks.append((plot_axis_angle_over_time, (p_time, p_quats), 
                               {"title_prefix": node_title, "filename": f"{node_prefix}_angles.png"}))
                               
            plot_tasks.append((plot_angular_velocity, (p_time, p_omega), 
                               {"title_prefix": node_title, "filename": f"{node_prefix}_omega.png"}))

    # Execute Plotting in Parallel
    print(f"[{datetime.now()}] Dispatching {len(plot_tasks)} plotting tasks to concurrent workers...")
    generated_images = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in plot_tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                generated_images.extend(future.result()) 
            except Exception as e:
                print(f"[{datetime.now()}] A plot worker encountered an error: {e}")
                import traceback
                traceback.print_exc()

    # Email Dispatch
    print(f"[{datetime.now()}] Plotting complete. Attached {len(generated_images)} files. Dispatching email...")
    msg = EmailMessage()
    msg['Subject'] = f"Piano Sim Run {run_id:03d} Completed - Plot Suite"
    msg['From'] = email_config['sender_email']
    msg['To'] = email_config['receiver_email']
    msg.set_content(f"Simulation {run_id:03d} plots attached. Memory-efficient parallel generation successful for {len(inspect_nodes)} nodes.")

    for img_path in generated_images:
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()
                # Use application/json for the stat files and image/png for the plots
                if str(img_path).endswith('.json'):
                    msg.add_attachment(img_data, maintype='application', subtype='json', filename=Path(img_path).name)
                else:
                    msg.add_attachment(img_data, maintype='image', subtype='png', filename=Path(img_path).name)
        except Exception as e:
            print(f"Could not attach {img_path}: {e}")

    try:
        with smtplib.SMTP_SSL(email_config['smtp_server'], email_config['smtp_port']) as smtp:
            smtp.login(email_config['sender_email'], email_config['sender_password'])
            smtp.send_message(msg)
        print(f"[{datetime.now()}] Email sent successfully.")
    except Exception as e:
        print(f"[{datetime.now()}] Failed to send email: {e}")

# ---------------------------------------------------------
# CLI Execution Wrapper
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print(f"[{datetime.now()}] Warning: 'python-dotenv' is not installed. Falling back to system environment variables.")

    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = os.environ.get("SMTP_PORT")

    if not sender_email or not sender_password or not receiver_email:
        print(f"[{datetime.now()}] FATAL ERROR: Email environment variables not set.")
        print("Please ensure SENDER_EMAIL, SENDER_PASSWORD, and RECEIVER_EMAIL are defined in your .env file.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("sim_path", type=str)
    args = parser.parse_args()
    target_dir = Path(args.sim_path).resolve()

    param_file = target_dir / "parameters.json"
    with open(param_file, "r") as f:
        metadata = json.load(f)

    # Config setup
    sim_config = metadata.get("config", {})
    inspect_nodes = sim_config.get("inspect_nodes", [64])
    if not inspect_nodes:
        inspect_nodes = [64]  # Safety fallback if list is empty

    rod = metadata["rod_derived"]
    email_config_dict = {
        "sender_email": sender_email,
        "sender_password": sender_password,
        "receiver_email": receiver_email,
        "smtp_server": smtp_server,
        "smtp_port": smtp_port,
    }

    process_and_plot(
        sim_path=target_dir,
        run_id=metadata.get("run_id", 0),
        config=sim_config,
        email_config=email_config_dict,
        m_node=rod["m_node"], dl=rod["dl"], 
        inertia=np.array(rod["inertia"]), K_se=np.array(rod["K_se"]), K_bt=np.array(rod["K_bt"]),
        inspect_nodes=inspect_nodes
    )
