import os, sys
from huggingface_hub import HfApi, hf_hub_download
SUBSETS = ["m4_daily","m4_hourly","m4_weekly","m4_monthly","m4_quarterly","m4_yearly","taxi_30min","uber_tlc_hourly",
           "wind_farms_hourly","wind_farms_daily","monash_pedestrian_counts","monash_kdd_cup_2018","mexico_city_bikes","ushcn_daily",
           "electricity_15min","monash_electricity_hourly","wiki_daily_100k","monash_temperature_rain","monash_rideshare","exchange_rate",
           "monash_traffic","monash_weather","monash_london_smart_meters","solar_1h","dominick","m5"]
api = HfApi(); info = api.dataset_info("autogluon/chronos_datasets", files_metadata=True)
files = [s for s in info.siblings if s.rfilename.split("/")[0] in SUBSETS and s.rfilename.endswith(".parquet")]
print(len(files), "files", sum(s.size or 0 for s in files)/1e9, "GB", flush=True)
root = os.path.expanduser("~/tsfm_rev/data/chronos_datasets")
for s in files:
    for attempt in range(5):
        try:
            hf_hub_download("autogluon/chronos_datasets", s.rfilename, repo_type="dataset", local_dir=root); break
        except Exception as e:
            print("retry", s.rfilename, e, flush=True)
    print("ok", s.rfilename, flush=True)
print("DONE", flush=True)
