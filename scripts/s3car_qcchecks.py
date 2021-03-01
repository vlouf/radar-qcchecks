import os
import sys
import glob
import argparse
import warnings
import traceback
from typing import Any, Dict, List

import netCDF4
import pandas as pd
import dask.bag as db
import radar_qcchecks


def buffer(func):
    """
    Decorator to catch and kill error message.
    """

    def wrapper(*args, **kwargs):
        try:
            rslt = func(*args, **kwargs)
        except OSError:
            print(f"File is invalid: {args}")
            return None
        except Exception:
            traceback.print_exc()
            return None
        return rslt

    return wrapper


@buffer
def check_file(infile: str) -> bool:
    """
    Check if file is empty.

    Parameter:
    ==========
    infile: str
        Input ODIM H5 file.

    Returns:
    ========
        True/False if file is not empty/empty.
    """
    if os.stat(infile).st_size == 0:
        return False
    else:
        return True


@buffer
def check_field(infile: str, field_name: str = "RHOHV") -> bool:
    """
    Check for the presence of the field fields in the ODIM h5 dataset.

    Parameter:
    ==========
    infile: str
        Input ODIM H5 file.

    Returns:
    ========
        True/False presence of the given field.
    """
    with netCDF4.Dataset(infile) as ncid:
        groups = ncid['/dataset1'].groups.keys()
        var = []
        for group in groups:
            if "data" not in group:
                continue
            name = ncid[f'/dataset1/{group}/what'].getncattr('quantity')
            var.append(name)

    if field_name in var:
        return True
    else:
        return False


def driver(infile: str) -> Any:
    """
    Buffer function to catch and kill errors about missing Sun hit.

    Parameters:
    ===========
    infile: str
        Input radar file.

    Returns:
    ========
    rslt: pd.DataFrame
        Pandas dataframe with the results from the solar calibration code.
    """
    try:
        rslt = radar_qcchecks.qccheck_radar_odim(infile, **FIELD_NAMES)
    except Exception:
        print(f"Problem with file {infile}.")
        traceback.print_exc()
        return None

    return rslt


def get_ground_radar_file(date: pd.Timestamp, rid: int) -> List:
    """
    Return the archive containing the radar file for a given radar ID and a
    given date.

    Parameters:
    ===========
    date: datetime
        Date.
    rid: int
        Radar ID number.

    Returns:
    ========
    flist: List
        Radar archive if it exists at the given date.
    """
    datestr = date.strftime("%Y%m%d")
    # Input directory checks.
    input_dir = os.path.join(VOLS_ROOT_PATH, str(rid), datestr)
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found/does not exist for radar {rid} at {datestr}.")
        return None

    flist = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if len(flist) == 0:
        print(f"No file found in {input_dir} for radar {rid} at {datestr}.")
        return None

    return flist


def mkdir(path: str) -> None:
    """
    Make directory if it does not already exist.
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    return None


def main():
    radar_infoset = pd.read_csv(CONFIG_FILE)
    for n in range(len(radar_infoset)):
        rid = radar_infoset.id[n]

        # Create output directories and check if output file exists
        outdir = os.path.join(OUTPUT_DATA_PATH, "qcdualpol")
        mkdir(outdir)
        outdir = os.path.join(outdir, rid)
        mkdir(outdir)
        output_filename = os.path.join(outdir, f"{rid}_{DATE}.csv")
        if os.path.exists(output_filename):
            print(f"Output file already exists for {rid} at {DATE}. Doing nothing.")
            continue

        # Get radar files for given date and RID.
        radar_flist = get_ground_radar_file(DTIME, rid)
        if radar_flist is None:
            continue

        # Check if this is a dual-polarisation radar.
        radar_flist = [f for f in radar_flist if check_file(f)]
        dualpol_files = [*map(check_field, radar_flist)]
        if not any(dualpol_files):
            print(f"The dual-polarization fields are not present for radar {rid}.")
            continue
        flist = [f for f, g in zip(radar_flist, dualpol_files) if g is True]
        print(f"Found {len(flist)} files with the uncorrected reflectivity for radar {rid} for date {DATE}.")

        # Processing files.
        bag = db.from_sequence(flist).map(driver)
        rslt = bag.compute()
        dataframe_list = [r for r in rslt if r is not None]
        if len(dataframe_list) == 0:
            print(f"No results for date {DATE}.")
            continue
        else:
            df = pd.DataFrame(dataframe_list[0], index=[0])
            for r in dataframe_list[1:]:
                df = df.append(r, ignore_index=True)
            df.set_index("time").to_csv(output_filename, float_format="%g")
            print(f"{output_filename} written.")

    return None


if __name__ == "__main__":
    VOLS_ROOT_PATH: str = "/srv/data/s3car-server/vols"
    CONFIG_FILE: str = "/srv/data/s3car-server/config/radar_site_list.csv"
    FIELD_NAMES: Dict = {
        "dbz_name": "DBZH",
        "zdr_name": "ZDR",
        "rhohv_name": "RHOHV",
        "phidp_name": "PHIDP",
        "kdp_name": "KDP",
    }

    parser_description = "Radar quality checks."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-d",
        "--date",
        dest="date",
        type=str,
        help="Value to be converted to Timestamp (str).",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output",
        default="/srv/data/s3car-server/",
        type=str,
        help="Directory for output data.",
    )

    args = parser.parse_args()
    RID = args.rid
    DATE = args.date
    OUTPUT_DATA_PATH = args.output
    try:
        DTIME = pd.Timestamp(DATE)
    except Exception:
        traceback.print_exc()
        sys.exit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
