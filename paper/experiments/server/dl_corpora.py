"""Download the corpus-audit sample: every subset of LOTSA (Salesforce/lotsa_data) and Time-300B (Maple728/Time-300B)
whose total size is below 250 MB, excluding climate reanalysis/simulation and cloud-trace subsets, plus Time-300B's
copy of Chronos's KernelSynth corpus (synthetic/training_corpus_kernel_synth_1m)."""
import os
from huggingface_hub import HfApi, hf_hub_download
R = os.path.expanduser("~/tsfm_rev/data"); api = HfApi()
EXCL = ("cmip6", "era5", "weatherbench", "azure", "alibaba", "largest_", "buildings", "tsmixup")
for repo, dest, depth in (("Salesforce/lotsa_data", "lotsa", 1), ("Maple728/Time-300B", "time300b", 2)):
    info = api.dataset_info(repo, files_metadata=True); groups = {}
    for s in info.siblings:
        parts = s.rfilename.split("/")
        if len(parts) <= depth: continue
        g = "/".join(parts[:depth]); groups.setdefault(g, []).append(s)
    keep = [g for g, fs in groups.items() if not any(e in g for e in EXCL) and sum(f.size or 0 for f in fs) < 250e6]
    if repo.startswith("Maple"): keep.append("synthetic/training_corpus_kernel_synth_1m")
    print(repo, len(keep), "subsets", sum(sum(f.size or 0 for f in groups[g]) for g in keep) / 1e9, "GB", flush=True)
    for g in keep:
        for f in groups[g]:
            for a in range(5):
                try:
                    hf_hub_download(repo, f.rfilename, repo_type="dataset", local_dir=os.path.join(R, dest)); break
                except Exception as e:
                    print("retry", f.rfilename, e, flush=True)
        print("ok", repo, g, flush=True)
print("DONE", flush=True)
