"""Read-only quota snapshot; avoid SDK2.2.3's duration parser failure."""
import json
from datetime import datetime, timezone
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest


class RawQuota:
    @classmethod
    def prepare_from(cls, response):
        data=response.json()
        return {k:data[k] for k in ['quotaRefreshTime','gpuQuota','tpuQuota'] if k in data}


if __name__=='__main__':
    api=KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as client:
        result=client.kernels.kernels_api_client._client.call(
            'kernels.KernelsApiService','GetAcceleratorQuotaStatistics',
            ApiGetAcceleratorQuotaStatisticsRequest(),RawQuota)
    print(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),**result},indent=2))
