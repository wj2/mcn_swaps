import argparse
import pickle
from datetime import datetime
import os

import mcn_swaps.task as mst
import mcn_swaps.training as mstr
import mcn_swaps.nets as msn
import mcn_swaps.analysis as msa

def create_parser():
    parser = argparse.ArgumentParser(
        description="train RNNs with different hyperparameters"
    )
    out_template = "swap-rnn_{model}_n{units}_s{sigma}_{jobid}.pkl"
    parser.add_argument(
        "-o",
        "--output_template",
        default=out_template,
        type=str,
        help="file to save the output in",
    )
    parser.add_argument(
        "--output_folder",
        default="../results/mcn_swaps/rnn_sweep/",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument("--jobid", default="0000", type=str)
    parser.add_argument("--model_type", default="RNN", type=str)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--sigma", default=8, type=float)
    parser.add_argument("--num_steps", default=2000, type=int)
    parser.add_argument("--n_samples", default=5000, type=int)
    parser.add_argument("--n_reps", default=10, type=int)
    return parser


model_type_dict = {
    "RNN": msn.SimpleRNN,
    "GRU": msn.SimpleGRU,
    "LSTM": msn.SimpleLSTM,
}


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.date = datetime.now()

    model_type = model_type_dict.get(args.model_type.upper())
    hidden_units = args.hidden_units
    out_dict = {
        "targets": [],
        "distractors": [],
        "responses": [],
    }

    for i in range(args.n_reps):    
        task = mst.RetrospectiveContinuousReportTask(sigma=args.sigma)
        model = mstr.make_model_for_task(msn.SimpleRNN, task, hidden_units)

        mstr.train_model_on_task(model, task, num_steps=args.num_steps)    

        targs, dists, resps = msa.sample_model_responses(
            task, model, n_samples=args.n_samples,
        )
        out_dict["targets"].append(targs)
        out_dict["distractors"].append(dists)
        out_dict["responses"].append(resps)

    out_dict = {k: np.stack(v, axis=0) for k, v in out_dict.items()}
    out_dict.update(vars(args))
    fn = args.output_template.format(
        model=args.model_type,
        units=args.hidden_units,
        sigma=args.sigma,
        jobid=args.jobid,
    )
    path = os.path.join(args.output_folder, fn)
    pickle.dump(out_dict, open(path, "wb"))
