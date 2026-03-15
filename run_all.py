import argparse

import baseline_lasso
import baseline_rf
import baseline_svr
import ann_model

HEADER = f"{'Method':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'Adj R2':<12}"
DIVIDER = '-' * len(HEADER)


def _print_row(name, m):
    print(f"{name:<20} {m['MSE']:.6f}  {m['RMSE']:.6f}  {m['MAE']:.6f}  {m['Adj_R2']:.6f}")


def main(data_path='../ccleaned_data.xlsx'):
    print(HEADER)
    print(DIVIDER)
    for name, fn in [
        ('Random Forest',  baseline_rf.train_and_evaluate),
        ('SVR',            baseline_svr.train_and_evaluate),
        ('LASSO',          baseline_lasso.train_and_evaluate),
        ('Our Work (ANN)', ann_model.train_and_evaluate),
    ]:
        results = fn(data_path)
        _print_row(name, results['combined'])
    print(DIVIDER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../ccleaned_data.xlsx')
    args = parser.parse_args()
    main(args.data)
