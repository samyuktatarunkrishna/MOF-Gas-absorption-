from ase.io import read
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
from pathlib import Path

def show_structure(refcode, save_dir="outputs/plots"):
    cif_path = Path("data/mof_structures") / f"{refcode}.cif"
    
    try:
        # Removed primitive_cell=True to avoid subtrans_included error
        atoms = read(str(cif_path))
    except Exception as e:
        print(f"[✘] Failed to read {refcode}: {e}")
        return

    fig = plt.figure(figsize=(5, 5))
    plot_atoms(atoms, ax=fig.add_subplot(111), radii=0.5, rotation=('45x,45y,0z'))
    plt.axis('off')
    
    save_path = Path(save_dir) / f"{refcode}_structure.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"[✔] Saved: {save_path}")
    return save_path