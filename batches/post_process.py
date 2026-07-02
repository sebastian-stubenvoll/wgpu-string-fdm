import numpy as np
from pathlib import Path
from datetime import datetime
import smtplib
from email.message import EmailMessage
import os
import pickle
import gzip
import concurrent.futures

# ---------------------------------------------------------
# Helper Functions 
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Parallel Worker Function for File Extraction
# ---------------------------------------------------------
def _process_single_chunk(args):
    nf, ef, m_node, inertia, K_se, K_bt = args
    
    # Pre-allocate dictionaries for full rod extraction
    chunk_nodes = {'pos': [], 'vel': [], 'mom': [], 'curv': []}
    chunk_edges = {'quat': [], 'ang_vel': [], 'force': [], 'strain': []}
                 
    t_ke, r_ke, bt_pe, ss_pe = [], [], [], []

    with gzip.open(nf, "rb") as f: n_chunk = pickle.load(f)
    with gzip.open(ef, "rb") as f: e_chunk = pickle.load(f)
        
    for i in range(len(n_chunk)):
        n_frame = n_chunk[i]
        e_frame = e_chunk[i]
        
        # --- FULL STATE EXTRACTION (Bypassing inhomogeneous shapes) ---
        # Nodes: [pos, vel, moment, curvature]
        chunk_nodes['pos'].append([n[0] for n in n_frame])
        chunk_nodes['vel'].append([n[1] for n in n_frame])
        chunk_nodes['mom'].append([n[2] for n in n_frame])
        chunk_nodes['curv'].append([n[3] for n in n_frame])
        
        # Edges: (orientation, [angular_vel, internal_force, strain])
        chunk_edges['quat'].append([e[0] for e in e_frame])
        chunk_edges['ang_vel'].append([e[1][0] for e in e_frame])
        chunk_edges['force'].append([e[1][1] for e in e_frame])
        chunk_edges['strain'].append([e[1][2] for e in e_frame])
        
        # --- GLOBAL ENERGY CALCULATION ---
        t_sum, bt_sum = 0, 0
        for j, node in enumerate(n_frame):
            weight = 0.5 if (j == 0 or j == len(n_frame)-1) else 1.0
            t_sum += E_T(node, m_node) * weight
            bt_sum += E_PB(node[3], K_bt) * weight
            
        r_sum, ss_sum = 0, 0
        for edge in e_frame:
            edge_quat, edge_vecs = edge
            edge_w, edge_f, edge_s = edge_vecs
            r_sum += E_R(edge_w, inertia)
            ss_sum += E_PS(edge_s, K_se)
            
        t_ke.append(t_sum)
        bt_pe.append(bt_sum)
        r_ke.append(r_sum)
        ss_pe.append(ss_sum)

    return (chunk_nodes, chunk_edges, t_ke, r_ke, bt_pe, ss_pe)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def process_and_email(sim_path, run_id, config, email_config, m_node, dl, inertia, K_se, K_bt):
    sim_path = Path(sim_path)
    node_files = sorted(sim_path.glob("n_*.pkl.gz"))
    edge_files = sorted(sim_path.glob("e_*.pkl.gz"))
    
    if hasattr(config, 'SIM_CONFIG'):
        sim_config = config.SIM_CONFIG
    else:
        sim_config = config
        
    print(f"[{datetime.now()}] Processing {len(node_files)} chunks for full rod history...")
    
    args_list = [(nf, ef, m_node, inertia, K_se, K_bt) for nf, ef in zip(node_files, edge_files)]
    
    t_ke_all, r_ke_all, bt_pe_all, ss_pe_all = [], [], [], []
    all_nodes = {'pos': [], 'vel': [], 'mom': [], 'curv': []}
    all_edges = {'quat': [], 'ang_vel': [], 'force': [], 'strain': []}
                 
    # 1. PARALLEL DATA EXTRACTION
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_single_chunk, args_list))
        
    for res in results:
        chunk_nodes, chunk_edges, t_ke, r_ke, bt_pe, ss_pe = res
        
        for key in all_nodes: all_nodes[key].extend(chunk_nodes[key])
        for key in all_edges: all_edges[key].extend(chunk_edges[key])
            
        t_ke_all.extend(t_ke)
        r_ke_all.extend(r_ke)
        bt_pe_all.extend(bt_pe)
        ss_pe_all.extend(ss_pe)

    # 2. DATA PACKAGING
    dt = sim_config["dt"]
    oversamp = sim_config["oversampling_factor"]
    time_full = np.arange(len(t_ke_all)) * (dt * oversamp)
    
    # Normalize shear potential energy to start at 0
    ss_pe_all = np.array(ss_pe_all, dtype=np.float32)
    ss_pe_all -= ss_pe_all[0]

    export_dict = {
        "time": time_full,
        "global_energy": {
            "kinetic_trans": np.array(t_ke_all, dtype=np.float32),
            "kinetic_rot": np.array(r_ke_all, dtype=np.float32),
            "potential_bend": np.array(bt_pe_all, dtype=np.float32),
            "potential_shear": ss_pe_all
        },
        # Convert packed lists into dense 3D arrays: (Time, Nodes/Edges, Dimension)
        "nodes": {k: np.array(v, dtype=np.float32) for k, v in all_nodes.items()},
        "edges": {k: np.array(v, dtype=np.float32) for k, v in all_edges.items()}
    }

    # Save to a single compressed file
    export_filename = sim_path / f"run_{run_id:03d}_full_timeseries.pkl.gz"
    print(f"[{datetime.now()}] Packaging and compressing data to {export_filename}...")
    with gzip.open(export_filename, "wb") as f:
        pickle.dump(export_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. EMAIL DISPATCH
    if hasattr(email_config, 'EMAIL_CONFIG'):
        e_config = email_config.EMAIL_CONFIG
    else:
        e_config = email_config

    if e_config and e_config.get('sender_email') and e_config.get('sender_password'):
        print(f"[{datetime.now()}] Sending email with attached full timeseries data...")
        msg = EmailMessage()
        msg['Subject'] = f"Piano Sim Run {run_id:03d} Completed - Full Kinematics"
        msg['From'] = e_config['sender_email']
        msg['To'] = e_config['receiver_email']
        
        file_size_mb = os.path.getsize(export_filename) / (1024 * 1024)
        msg.set_content(f"Simulation {run_id:03d} finished processing.\n\nExtracted full spatial data is attached ({file_size_mb:.2f} MB).\n\nLoad it in Jupyter using:\nimport gzip, pickle\nwith gzip.open('{export_filename.name}', 'rb') as f:\n    data = pickle.load(f)\n\n# Example shapes:\n# data['nodes']['pos'].shape -> (N_time, 128, 3)\n# data['edges']['quat'].shape -> (N_time, 128, 4)")

        with open(export_filename, 'rb') as f:
            msg.add_attachment(
                f.read(), 
                maintype='application', 
                subtype='gzip', 
                filename=export_filename.name
            )
            
        try:
            with smtplib.SMTP_SSL(e_config['smtp_server'], e_config['smtp_port']) as smtp:
                smtp.login(e_config['sender_email'], e_config['sender_password'])
                smtp.send_message(msg)
            print(f"[{datetime.now()}] Email sent successfully.")
        except Exception as e:
            print(f"[{datetime.now()}] Failed to send email: {e}")



