from ebo_core.bo import bo


class MyPool(object):
    def __init__(self, a, b, useAzure, thresAzure):
        if useAzure:
            from azure_tools.azurepool import AzurePool
            self.pool = AzurePool(a, b)
        self.useAzure = useAzure
        self.thresAzure = thresAzure

    def map(self, parameters, job_id, really_useAzure=True):
        if really_useAzure and len(parameters) > self.thresAzure and self.useAzure:
            return self.pool.map(parameters, job_id)
        res = []
        for p in parameters:
            b = bo(*p)
            res.append(b.run())
        return res

    def end(self):
        if self.useAzure:
            self.pool.end()

    def delete_containers(self):
        if self.useAzure:
            self.pool.delete_containers()
