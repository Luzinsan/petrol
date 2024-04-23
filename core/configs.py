from os.path import join, exists
from pathlib import Path

COLORMAP = [[0.0, '#3f7f93'],
            [0.1, '#6397a7'],
            [0.2, '#88b1bd'],
            [0.3, '#acc9d2'],
            [0.4, '#d1e2e7'],
            [0.5, '#f2f2f2'],
            [0.6, '#f6cdd0'],
            [0.7, '#efa8ad'],
            [0.8, '#e8848b'],
            [0.9, '#e15e68'],
            [1.0, '#da3b46']]

PLOT_THEME='plotly_dark'
# PLOT_THEME='none'
VERBOSE=False
SHOW_PLOTS=False
APPEND_TO_EXISTS=False
DROP_OLD_COLUMNS=True
TARGET='Добыча воды за 2 ч ,м3, лаг -1'
TIME_AXIS='YY-MM-DD HH:00'
INSERT_NEARBY=True
LOG=(False, False) # log_x=LOG[0], log_y=LOG[1]
TEXTFONT_SIZE=10
FILEPATH=None

ROOT = Path('/home/prog3/notebooks/petrol/')
DATA = join(ROOT, 'data')
PLOTS = join(ROOT, 'plots')
MODELS = join(ROOT, 'models')