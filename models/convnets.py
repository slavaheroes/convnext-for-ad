import monai


def make_densenet1213d(spatial_dims=3,
                       n_input_channels=1,
                       num_classes=2):
    
    return monai.networks.nets.densenet121(spatial_dims=spatial_dims, 
                                           in_channels=n_input_channels, 
                                           out_channels=num_classes)


def make_resnet103d(spatial_dims=3, 
                    n_input_channels=1, 
                    num_classes=2):
    return monai.networks.nets.resnet10(spatial_dims=spatial_dims, 
                                          n_input_channels=n_input_channels, 
                                          num_classes=num_classes)
    
def make_resnet183d(spatial_dims=3, 
                    n_input_channels=1, 
                    num_classes=2):
    return monai.networks.nets.resnet18(spatial_dims=spatial_dims, 
                                          n_input_channels=n_input_channels, 
                                          num_classes=num_classes)

def make_resnet343d(spatial_dims=3, 
                    n_input_channels=1, 
                    num_classes=2):
    return monai.networks.nets.resnet34(spatial_dims=spatial_dims, 
                                          n_input_channels=n_input_channels, 
                                          num_classes=num_classes)


def make_resnet1013d(spatial_dims=3, 
                    n_input_channels=1, 
                    num_classes=2):
    return monai.networks.nets.resnet101(spatial_dims=spatial_dims, 
                                          n_input_channels=n_input_channels, 
                                          num_classes=num_classes)

def make_resnet1523d(spatial_dims=3, 
                    n_input_channels=1, 
                    num_classes=2):
    return monai.networks.nets.resnet152(spatial_dims=spatial_dims, 
                                          n_input_channels=n_input_channels, 
                                          num_classes=num_classes)
