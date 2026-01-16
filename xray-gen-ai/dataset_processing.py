import fiftyone as fo
import fiftyone.types as fot
import fiftyone.core.labels as fol  # 👈 important

data_path = "/home/ssy/Desktop/dataset/Color"

dataset = fo.Dataset.from_dir(
    dataset_dir=data_path,
    dataset_type=fot.ImageDirectory,
    name="xray_dataset",
    overwrite=True,
)

print("Number of samples:", len(dataset))


# Launch app
session = fo.launch_app(
    dataset,
    address="localhost",
    port=5151,
    auto=True,
)

session.wait()
