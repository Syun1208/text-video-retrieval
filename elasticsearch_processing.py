from typing import Any
from elasticsearch import Elasticsearch


class ElasticSeach:
    def __init__(self):
        self.client = Elasticsearch(hosts=['http://0.0.0.0:9200'])

    
    def set_up(self, es_pwd='fzO062HMgWRLoiFuSe05jDft', cloud_id='aic2023:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyRlOGRhNmQ3ODBmZDk0OTFiOTI2NDg3MGU2MmY4OWJiNyRiOGE4MTVjODRmNWM0MzliOTk3N2E3MGFmZWE1MTM3Zg=='):
        ELASTIC_PASSWORD = es_pwd

        # Found in the 'Manage Deployment' page
        CLOUD_ID = cloud_id

        # Create the client instance
        self.client = Elasticsearch(
            cloud_id=CLOUD_ID,
            basic_auth=("elastic", ELASTIC_PASSWORD)
        )

        # Successful response!
        self.client.info()


    def image_search(self, image_id, k):
        pass

    def text_search(self, text, k):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

def main():
    es = ElasticSeach()
    es.set_up()
    print(es.client.info())

if __name__ == '__main__':
    main()