# HSBC Under-35 Customer Segmentation

Segments 48,805 HSBC customers aged under 35 (synthetic data, HKD) into three personas:
**Established Wealth Builders** (46%), **Digital-Native Starters** (32%) and **Underserved Low-Balance** (22%).

Steps: data audit → cleaning → KNN imputation → outlier checks → log scaling →
K-Means / Ward / FAMD+GMM → consensus ensemble → bootstrap and permutation validation → personas.

## Structure

~~~
segmentation.py      # full pipeline
requirements.txt     # dependencies
ST138D-XLS-ENG.xlsx  # input data
figures/             # generated charts
outputs/             # generated tables and results
~~~

## Clone and run

~~~bash
git clone https://github.com/YOUR_USERNAME/hsbc-under35-segmentation.git
cd hsbc-under35-segmentation

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python segmentation.py
~~~

Results are written to `figures/` and `outputs/`. Requires Python 3.10+.

*HKUST Business School · Business Analytics*
