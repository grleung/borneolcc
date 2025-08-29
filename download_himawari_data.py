"""
This script downloads himawari data using the himawari_api.
"""

import datetime as dt
import himawari_api as hapi

# save folder
base_dir = "/squall/gleung/ahi/"

# start and end time
start_time = dt.datetime(2019, 9, 16, 20, 00)
end_time = dt.datetime(2019, 9, 19, 20, 00)

# Should not need to edit anything below this


protocol = "s3"
fs_args = {}
satellite = "himawari-8"
product_level = "L1b"
product = "Rad"

# pick sector and channels
sector = "FLDK"
scene_abbr = None  # None download and find both locations
channels = None  # select all channels
channels = [
    "B01",
    "B02",
    "B03",
    "B04",
]  # select channels subset
filter_parameters = {}
filter_parameters["channels"] = channels
filter_parameters["scene_abbr"] = scene_abbr

bucket_fpaths = hapi.find_files(
    protocol=protocol,
    fs_args=fs_args,
    satellite=satellite,
    product_level=product_level,
    product=product,
    sector=sector,
    start_time=start_time,
    end_time=end_time,
    filter_parameters=filter_parameters,
    connection_type="bucket",
    base_dir=None,
    group_by_key=None,
    verbose=False,
)
n_threads = 20  # n_parallel downloads
force_download = False  # whether to overwrite existing data on disk

fpaths = hapi.download_files(
    base_dir=base_dir,
    protocol=protocol,
    fs_args=fs_args,
    satellite=satellite,
    product_level=product_level,
    product=product,
    sector=sector,
    start_time=start_time,
    end_time=end_time,
    filter_parameters=filter_parameters,
    n_threads=n_threads,
    force_download=force_download,
    check_data_integrity=True,
    progress_bar=True,
    verbose=True,
)
