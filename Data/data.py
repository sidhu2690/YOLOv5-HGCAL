import ROOT as rt
import numpy as np
import os


def create_directories(base_dir='data'):
    for split in ['train', 'val']:
        os.makedirs(f"{base_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{base_dir}/{split}/labels", exist_ok=True)


def log_scale_adc(img, adc_max=1_000_000):
    """255 * log(1 + ADC) / log(1 + ADC_max)"""
    return (255.0 * np.log1p(np.maximum(img, 0)) / np.log1p(adc_max)).astype(np.uint8)


def collapse_layers(img):
    """47 layers → 16 channels (15 groups of 3 + 1 group of 2)"""
    out = np.zeros((736, 736, 16), dtype=np.float32)
    for ch in range(15):
        out[:, :, ch] = img[:, :, ch*3] + img[:, :, ch*3 + 1] + img[:, :, ch*3 + 2]
    out[:, :, 15] = img[:, :, 45] + img[:, :, 46]
    return out


def process_root_file(in_file, particle, start_index=0, n_events=-1,
                      val_fraction=0.2, box_size='n', base_dir='data', adc_max=1_000_000):
    
    # 0=electron, 1=muon-, 2=photon, 3=positron
    cls = {'ele': 0, 'muon': 1, 'photon': 2, 'pos': 3}[particle]
    dim1, dim2 = 736, 736
    
    create_directories(base_dir)
    
    infile = rt.TFile.Open(in_file, "READ")
    hitTree = infile.Get('Eta_Phi_CellWiseSegmentation')
    labelTree = infile.Get('YOLOLabels')
    
    t_events = labelTree.GetEntriesFast()
    if n_events <= 0 or n_events > t_events:
        n_events = t_events
    n_train = int(n_events * (1.0 - val_fraction))
    
    print(f"Processing {n_events} events ({n_train} train, {n_events - n_train} val)")
    
    # Build hit index: event_id → [(ieta, iphi, layer, ADC), ...]
    print("Building hit index...")
    hit_index = {i: [] for i in range(n_events)}
    for j in range(hitTree.GetEntries()):
        hitTree.GetEntry(j)
        evt = hitTree.event_id
        if evt < n_events:
            hit_index[evt].append((hitTree.ieta, hitTree.iphi, hitTree.layer, hitTree.ADC))
    
    # Process each event
    print("Saving events...")
    for evt in range(n_events):
        
        split = "train" if evt < n_train else "val"
        filename = f"{particle}_{start_index + evt:05d}"
        
        # Save label
        labelTree.GetEntry(evt)
        x_center = labelTree.iphi_seed / dim2
        y_center = labelTree.ieta_seed / dim1
        
        f_eta = {'s': labelTree.f90_eta, 'n': labelTree.f95_eta, 'l': labelTree.f98_eta}[box_size]
        f_phi = {'s': labelTree.f90_phi, 'n': labelTree.f95_phi, 'l': labelTree.f98_phi}[box_size]
        width = (2 * f_phi + 1) / dim2
        height = (2 * f_eta + 1) / dim1
        
        with open(f"{base_dir}/{split}/labels/{filename}.txt", 'w') as f:
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Build image
        img = np.zeros((dim1, dim2, 47), dtype=np.float32)
        for ieta, iphi, layer, ADC in hit_index[evt]:
            if 0 <= ieta < dim1 and 0 <= iphi < dim2 and 1 <= layer <= 47:
                img[ieta, iphi, layer - 1] += ADC
        
        # Collapse 47 → 16, log scale, save
        img = collapse_layers(img)
        img = log_scale_adc(img, adc_max)
        np.save(f"{base_dir}/{split}/images/{filename}.npy", img)
        
        if (evt + 1) % 100 == 0:
            print(f"  {evt + 1}/{n_events}")
    
    infile.Close()
    print(f"✅ Done! Next index: {start_index + n_events}")
    return start_index + n_events

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROOT → YOLO data pipeline")
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--particle", required=True,
                        choices=["ele", "muon", "photon", "pos"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--n_events", type=int, default=-1)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--box_size", choices=["s", "n", "l"], default="n")
    parser.add_argument("--base_dir", default="data")
    args = parser.parse_args()

    process_root_file(
        in_file=args.in_file,
        particle=args.particle,
        start_index=args.start_index,
        n_events=args.n_events,
        val_fraction=args.val_fraction,
        box_size=args.box_size,
        base_dir=args.base_dir,
    )

