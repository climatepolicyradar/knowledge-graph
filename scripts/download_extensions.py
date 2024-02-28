import shutil
from pathlib import Path
import httpx
import html5lib
from tqdm import tqdm
import logging
import tarfile

log = logging.getLogger("rich")

extensions = [
    "Echo",
    "MobileFrontend",
    "RevisionSlider",
    "TwoColConflict",
    "WikiEditor",
    "WikibaseQualityConstraints",
]

BASE_URL = "https://extdist.wmflabs.org/dist/extensions/"
# this page is big list of extensions in a directory which looks something like this:
#
# Index of /dist/extensions/
# ../
# 3D-REL1_35-a28184e.tar.gz                          18-Dec-2023 08:00              341017
# 3D-REL1_36-c7ba35d.tar.gz                          22-May-2022 06:00              340097
# 3D-REL1_37-12fefc3.tar.gz                          25-Oct-2022 08:00              335178
# 3D-REL1_38-a1d492e.tar.gz                          04-May-2023 20:00              341290
# 3D-REL1_39-ca3874c.tar.gz                          19-Feb-2024 07:00              324582
# 3D-REL1_40-31d8057.tar.gz                          13-Feb-2024 07:00              326731
# 3D-REL1_41-d830cdb.tar.gz                          14-Feb-2024 07:00              328329
# 3D-master-f0d428f.tar.gz                           14-Feb-2024 08:00              338397
# AControlImageLink-REL1_35-ba39f13.tar.gz           23-Oct-2022 01:00               23950
# AControlImageLink-REL1_36-3ae4d44.tar.gz           22-May-2022 06:00               57111
# AControlImageLink-REL1_37-013facb.tar.gz           23-Oct-2022 01:00               57003
# AControlImageLink-REL1_38-2db8239.tar.gz           01-Jan-2023 01:00               74152
# AControlImageLink-REL1_39-8edd352.tar.gz           04-Feb-2024 20:00               64006
# AControlImageLink-REL1_40-b1cbd45.tar.gz           04-Feb-2024 20:00               64031
# AControlImageLink-REL1_41-59cc733.tar.gz           04-Feb-2024 20:00               66015
# AControlImageLink-master-3b2d0ef.tar.gz            10-Feb-2024 06:00               73117
# AJAXPoll-REL1_35-e288f5b.tar.gz                    11-Dec-2023 08:00              166291
# AJAXPoll-REL1_36-dfb5c6d.tar.gz                    31-May-2022 06:00              160721
# AJAXPoll-REL1_37-155635a.tar.gz                    29-Nov-2022 07:00              160163
# AJAXPoll-REL1_38-db3c48b.tar.gz                    17-May-2023 06:00              167879
# AJAXPoll-REL1_39-dded7b8.tar.gz                    15-Feb-2024 23:00              151379
# AJAXPoll-REL1_40-790bb52.tar.gz                    06-Feb-2024 07:00              151944
# AJAXPoll-REL1_41-76d4fd2.tar.gz                    07-Feb-2024 07:00              153395
# AJAXPoll-master-57228a3.tar.gz                     10-Feb-2024 06:00              161596
# AbsenteeLandlord-REL1_35-ebafcba.tar.gz            13-Nov-2023 08:00               43575
# AbsenteeLandlord-REL1_36-f422b39.tar.gz            24-May-2022 06:00               68981
# AbsenteeLandlord-REL1_37-98fb8bd.tar.gz            15-Nov-2022 07:00               69532
# AbsenteeLandlord-REL1_38-c75375f.tar.gz            01-Feb-2023 07:00               87200
# AbsenteeLandlord-REL1_39-00809ee.tar.gz            12-Feb-2024 07:00               79056
# AbsenteeLandlord-REL1_40-6d93ef7.tar.gz            13-Feb-2024 07:00               79095
# AbsenteeLandlord-REL1_41-f2f587d.tar.gz            14-Feb-2024 07:00               81593
# AbsenteeLandlord-master-63c4061.tar.gz             08-Feb-2024 09:00               81603
# ...

# we want to download the latest version of each extension and save an unzipped version to
# the extensions directory

extensions_dir = Path("./wikibase/extensions")


# recursively remove all files and directories in the extensions directory
def recursively_remove_dir(dir):
    if dir.exists():
        for item in dir.iterdir():
            if item.is_dir():
                recursively_remove_dir(item)
            else:
                item.unlink()
        dir.rmdir()


recursively_remove_dir(extensions_dir)

# create the extensions directory if it doesn't exist
extensions_dir.mkdir(exist_ok=True, parents=True)

with httpx.Client() as client:
    response = client.get(BASE_URL)
    parsed_extensions_page = html5lib.parse(response.text, namespaceHTMLElements=False)
    all_links = [link.text for link in parsed_extensions_page.findall(".//a")]

progress_bar = tqdm(extensions, desc="Downloading extensions")
for extension in progress_bar:
    # We want to choose version number which corresponds to mediawiki version 1.39
    # so the name should contain "REL1_39"
    extension_options = [
        link
        for link in all_links
        if link.lower().startswith(extension.lower() + "-")
        and "master" not in link.lower()
    ]
    try:
        version_name = next(
            option
            for option in extension_options
            if "REL1_39" in option and option.endswith(".tar.gz")
        )
    except ValueError:
        log.error(
            f"Couldn't find {extension} in the list of extensions available at {BASE_URL}"
        )
        continue

    # download the latest version of the extension
    url = BASE_URL + version_name
    with httpx.Client() as client:
        response = client.get(url)
        with open(extensions_dir / version_name, "wb") as f:
            f.write(response.content)

    with tarfile.open(extensions_dir / version_name, "r:gz") as tar:
        # extract to a temporary directory
        # move it to the extensions directory, renamed according to the original list
        # remove the temporary directory
        tar.extractall(extension_tmp_dir := extensions_dir / "tmp")
        shutil.move(extension_tmp_dir / tar.getnames()[0], extensions_dir / extension)

    # remove the tar file
    (extensions_dir / version_name).unlink()

recursively_remove_dir(extension_tmp_dir)
