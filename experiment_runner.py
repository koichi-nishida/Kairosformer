#!/usr/bin/env python3
import subprocess, time, os, csv, re, shlex, datetime, pathlib

# ----------------------------- USER CONFIG -----------------------------
# Models to run
MODELS = [
    "Kairosformer_A1",
    "Kairosformer_A2",
    "Kairosformer_A3",
    "Kairosformer_v0",
    "Autoformer",
    "Informer",
    "Transformer",
]

# Prediction lengths to sweep
PRED_LENS = [24, 48, 96, 192, 336, 720]

# Common hyperparams (you can edit globally here)
# COMMON = dict(
#     is_training=1,
#     model_id="test",
#     data="ETTh1",
#     root_path="./data/ETT/",
#     data_path="ETTh1.csv",
#     features="M",
#     target="OT",
#     freq="h",
#     seq_len=96,
#     label_len=48,
#     enc_in=7,
#     dec_in=7,
#     c_out=7,
#     d_model=512,
#     n_heads=8,
#     e_layers=2,
#     d_layers=1,
#     d_ff=2048,
#     moving_avg=25,
#     factor=1,
#     dropout=0.05,
#     embed="timeF",
#     activation="gelu",
#     learning_rate=1e-4,
#     train_epochs=10,
#     batch_size=32,
#     itr=1,
#     des="grid",
# )

COMMON = dict(
    is_training=1,
    root_path="./data/electricity/",
    data_path="electricity.csv",
    model_id="ECL_96_96",
    model="Autoformer",
    data="custom",
    features="M",
    seq_len=96,
    label_len=48,
    pred_len=96,
    e_layers=2,
    d_layers=1,
    factor=3,
    enc_in=321,
    dec_in=321,
    c_out=321,
    des="Exp",
    itr=1,
)


# Datasets registry (customizable). Add more blocks as needed.
DATASETS = {
    # name: overrides to COMMON
    # "ETTh1": dict(
    #     data="ETTh1", root_path="./data/ETT/", data_path="ETTh1.csv",
    #     features="M", target="OT", freq="h", enc_in=7, dec_in=7, c_out=7
    # ),
    # "ETTh2": dict(
    #     data="ETTh2", root_path="./data/ETT/", data_path="ETTh2.csv",
    #     features="M", target="OT", freq="h", enc_in=7, dec_in=7, c_out=7
    # ),
    # "ETTm1": dict(
    #     data="ETTm1", root_path="./data/ETT/", data_path="ETTm1.csv",
    #     features="M", target="OT", freq="t", enc_in=7, dec_in=7, c_out=7
    # ),
    # "ETTm2": dict(
    #     data="ETTm2", root_path="./data/ETT/", data_path="ETTm2.csv",
    #     features="M", target="OT", freq="t", enc_in=7, dec_in=7, c_out=7
    # ),
    # Add your own datasets here, e.g. Electricity, Exchange, Weather, etc.
    "Electricity": dict(data="custom", root_path="./data/electricity/", data_path="electricity.csv", 
                        features="M", target="OT", freq="h", enc_in=321, dec_in=321, c_out=321),
    "Exchange": dict(data="custom", root_path="./data/exchange_rate/", data_path="exchange_rate.csv",
                        features="M", target="OT", freq="d", enc_in=8, dec_in=8, c_out=8),
    "Traffic": dict(data="custom", root_path="./data/traffic/", data_path="traffic.csv",
                        features="M", target="OT", freq="h", enc_in=862, dec_in=862, c_out=862, train_epochs=3),
    "Weather": dict(data="custom", root_path="./data/weather/", data_path="weather.csv",
                        features="M", target="OT", freq="h", enc_in=21, dec_in=21, c_out=21, train_epochs=2),

}

# Select which datasets to run this sweep on:
SELECTED_DATASETS = ["Weather"]  # change to e.g. ["ETTh1","ETTh2","ETTm1"]
# ----------------------------------------------------------------------

# Regexes to parse metrics from run.py stdout (be lenient to ordering/spacing)
RE_MSE = re.compile(r"\bmse\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
RE_MAE = re.compile(r"\bmae\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def dict_to_cli(d):
    parts = []
    for k, v in d.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        else:
            parts.append(f"{flag} {v}")
    return " ".join(map(str, parts))

def build_command(model, pred_len, dataset_name):
    # merge COMMON + dataset overrides + model + pred_len
    args = COMMON.copy()
    args.update(DATASETS[dataset_name])
    args["model"] = model
    args["pred_len"] = pred_len
    cli = f"python -u run.py {dict_to_cli(args)}"
    return cli, args

def parse_metrics(stdout):
    # try to get the last occurrence (sometimes printed multiple times)
    mse_matches = RE_MSE.findall(stdout)
    mae_matches = RE_MAE.findall(stdout)
    mse = float(mse_matches[-1]) if mse_matches else float("nan")
    mae = float(mae_matches[-1]) if mae_matches else float("nan")
    return mse, mae

def write_csv_row(csv_path, row, header):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logs_dir = "logs"
    base_csv_dir = "csv_data"
    ensure_dir(logs_dir)
    ensure_dir(base_csv_dir)

    for model in MODELS:
        model_csv_dir = os.path.join(base_csv_dir, model)
        model_logs_dir = os.path.join(logs_dir, model)
        ensure_dir(model_csv_dir)
        ensure_dir(model_logs_dir)

        for pred_len in PRED_LENS:
            # One CSV per pred_len for this model, aggregated over datasets/runs
            csv_path = os.path.join(model_csv_dir, f"pred_{pred_len}.csv")
            header = [
                "timestamp", "model", "dataset", "pred_len",
                "mse", "mae", "total_time_s",
                "cmdline"
            ]

            for dataset in SELECTED_DATASETS:
                cmd, args = build_command(model, pred_len, dataset)
                print(f"\n[RUN] {dataset} | {model} | pred_len={pred_len}")
                print(cmd)

                t0 = time.perf_counter()
                # Use shell=False with shlex.split for safety
                proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
                t1 = time.perf_counter()
                total_time = round(t1 - t0, 4)

                # Save stdout/stderr logs for reproducibility
                log_name = f"{dataset}_pred{pred_len}_{timestamp}.log"
                with open(os.path.join(model_logs_dir, log_name), "w") as lf:
                    lf.write("=== COMMAND ===\n")
                    lf.write(cmd + "\n\n")
                    lf.write("=== STDOUT ===\n")
                    lf.write(proc.stdout or "")
                    lf.write("\n=== STDERR ===\n")
                    lf.write(proc.stderr or "")

                mse, mae = parse_metrics(proc.stdout + "\n" + proc.stderr)
                row = dict(
                    timestamp=timestamp,
                    model=model,
                    dataset=dataset,
                    pred_len=pred_len,
                    mse=mse,
                    mae=mae,
                    total_time_s=total_time,
                    cmdline=cmd,
                )
                write_csv_row(csv_path, row, header)

                # Also echo a compact line to console
                status = "OK" if proc.returncode == 0 else f"RC={proc.returncode}"
                print(f"[DONE] {dataset} | {model} | pred={pred_len} | mse={mse:.6f} | mae={mae:.6f} | time={total_time:.2f}s | {status}")

if __name__ == "__main__":
    main()
