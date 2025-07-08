class BaseRenderer:
    def render(self,**kwargs):
        raise NotImplementedError
    
    def verify_format(self,sample,selected_columns,name):
        raise NotImplementedError
    
    def get_values(self,sample,key):
        #functions for getting nested columns

        keys = key.split(".")
        for k in keys:
            if isinstance(sample,dict) and k in sample:
                sample = sample[k]
            else:
                return None
        return sample