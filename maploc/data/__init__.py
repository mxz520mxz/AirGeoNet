from .huawei.dataset import HuaweiDataModule
from .VPAir.dataset import VPAirDataModule
from .Remote_Data.dataset import RemoteDataModule
from .ALL.dataset import ALLDataModule

modules = {"huawei":HuaweiDataModule, "VPAir":VPAirDataModule, "Remote":RemoteDataModule, "ALL":ALLDataModule}
