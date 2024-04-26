from typing import Any
import tempfile
import lindi
from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
import sortingview.views as sv
import dandi.dandiarchive as da

# This script generated the following figurl on 4/26/2024
# https://figurl.org/f?v=npm://@fi-sci/figurl-sortingview@12/dist&d=sha1://bad46be226722ed7b7b3dac047dbc1433afee1d4&label=Falcon%20Benchmark%20M1%3A%20EMG%20Summary

# DANDISET 000941
# https://neurosift.app/?p=/dandiset&dandisetId=000941&dandisetVersion=draft
# https://dandiarchive.org/dandiset/000941/draft


# Example
# https://neurosift.app/?p=/nwb&url=https://api.dandiarchive.org/api/assets/ead69afc-0c2b-4082-9d06-e7650b00eca1/download/&dandisetId=000941&dandisetVersion=draft

# Based on code from:
# https://github.com/snel-repo/falcon-challenge/blob/main/data_demos/m1.ipynb
# See snel_repo_falcon_challenge_license.txt


def falcon_benchmark_m1_emg_summary():
    dandiset_id = '000941'
    parsed_url = da.parse_dandi_url(f"https://dandiarchive.org/dandiset/{dandiset_id}")
    with parsed_url.navigate() as (client, dandiset, assets):
        if dandiset is None:
            print(f"Dandiset {dandiset_id} not found.")
            return

        session_views = []
        for asset_obj in dandiset.get_assets('path'):
            if not asset_obj.path.endswith(".nwb"):
                continue
            asset = {
                "identifier": asset_obj.identifier,
                "path": asset_obj.path,
                "size": asset_obj.size,
                "download_url": asset_obj.download_url,
                "dandiset_id": dandiset_id,
            }
            print(f'Processing {asset["path"]}')
            session_url = asset['download_url']
            session_view = generate_session_view(session_url=session_url)
            session_views.append((asset['path'], session_view))
            print('')
        main_view = sv.TabLayout(
            items=[
                sv.TabLayoutItem(
                    label=x[0],
                    view=x[1]
                )
                for x in session_views
            ],
            tab_bar_layout='vertical'
        )
        print('Generating figure for Falcon Benchmark M1: EMG Summary')
        figurl = main_view.url(label='Falcon Benchmark M1: EMG Summary')
        print(figurl)


def generate_session_view(session_url: str, ):
    # some trialized EMG separated by muscle for a single session

    f = lindi.LindiH5pyFile.from_hdf5_file(session_url, local_cache=lindi.LocalCache())
    io = NWBHDF5IO(file=f, mode='r')
    nwbfile: Any = io.read()

    trial_info = (
        nwbfile.trials.to_dataframe()
        .reset_index()
        .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
    )
    muscles = [ts for ts in nwbfile.acquisition['preprocessed_emg'].time_series]
    raw_emg = nwbfile.acquisition['preprocessed_emg']
    emg = np.vstack([raw_emg.get_timeseries(m).data[:] for m in muscles]).T
    time = raw_emg.get_timeseries(muscles[0]).timestamps[:]

    condition_views = []
    condition_ids = sorted(trial_info['condition_id'].unique())
    for condition_id in condition_ids:
        condition_trials = trial_info.loc[trial_info['condition_id'] == condition_id]
        print(f'Condition {condition_id} has {condition_trials.shape[0]} trials')
        condition_trials = trial_info.loc[trial_info['condition_id'] == condition_id]
        fig, ax = plt.subplots(len(muscles) // 2, 2, figsize=(10, 14), sharex=True, sharey=True)
        axs = ax.flatten()

        resamp_time = np.arange(0, time[-1], 0.02)

        for trial in range(condition_trials.shape[0]):
            tr = condition_trials.iloc[trial]
            # get the timestamps between start and stop time
            start = tr['start_time']
            stop = tr['end_time']

            start_idx = np.where((resamp_time >= start - 0.01) & (resamp_time <= start + 0.01))[0]
            if len(start_idx) == 0:
                print(f'Warning: start time {start} not found in timestamps for trial {trial}')
                start_idx = None
            else:
                start_idx = start_idx[0]
            stop_idx = np.where((resamp_time >= stop - 0.01) & (resamp_time <= stop + 0.01))[0]
            if len(stop_idx) == 0:
                print(f'Warning: stop time {stop} not found in timestamps for trial {trial}')
                stop_idx = None
            else:
                stop_idx = stop_idx[0]
            if start_idx is not None and stop_idx is not None:
                for i, muscle in enumerate(muscles):
                    signal = emg[start_idx:stop_idx, i]
                    t = np.linspace(0, stop - start, len(signal))
                    axs[i].plot(t, signal, label=muscle, alpha=0.7, linewidth=0.75, color='k')
                    axs[i].set_ylabel(muscle)

        ax[-1, 0].set_xlabel('Time (s)')
        ax[-1, 1].set_xlabel('Time (s)')
        plt.suptitle(f'Single Trial EMG, Condition {condition_id}')
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_fname = f'{tmpdir}/plot.jpg'
            plt.savefig(plot_fname, dpi=300)
            vv = sv.Image(image_path=plot_fname)
            condition_views.append((
                'Condition ' + str(condition_id),
                vv
            ))
        plt.close(fig)
    return sv.TabLayout(
        items=[
            sv.TabLayoutItem(
                label=x[0],
                view=x[1]
            )
            for x in condition_views
        ],
        tab_bar_layout='vertical'
    )


if __name__ == '__main__':
    falcon_benchmark_m1_emg_summary()
