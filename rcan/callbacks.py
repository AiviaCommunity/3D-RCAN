# Copyright 2021 SVision Technologies LLC.
# Copyright 2021-2022 Leica Microsystems, Inc.
# Creative Commons Attribution-NonCommercial 4.0 International Public License
# (CC BY-NC 4.0) https://creativecommons.org/licenses/by-nc/4.0/

import functools
import tqdm

from tqdm.utils import IS_WIN
from tqdm.keras import TqdmCallback as _TqdmCallback


class TqdmCallback(_TqdmCallback):
    def __init__(self):
        super().__init__(
            tqdm_class=functools.partial(
                tqdm.tqdm, dynamic_ncols=True, ascii=IS_WIN))
        self.on_batch_end = self.bar2callback(
            self.batch_bar, pop=['batch', 'size'])
