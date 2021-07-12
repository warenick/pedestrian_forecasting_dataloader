# Pedestrian Forecasting Dalaoader

## Supported datasets:

    * ETH
    * UCY
    * SDD 


## downloading dataset:
    
    python download_dataset.py --path_to_folder=./data/

## Usage example: 

    from config import cfg
    path = "./"
    cfg["one_ped_one_traj"] = False
    cfg["raster_params"]["use_segm"] = True
    
    
    files = [
             "biwi_eth/biwi_eth.txt",
             "eth_hotel/eth_hotel.txt",
             "UCY/zara02/zara02.txt",
             "UCY/zara01/zara01.txt",
             "UCY/students01/students01.txt",
             "UCY/students03/students03.txt",
             ]
    dataset = DatasetFromTxt(path, files, cfg_=cfg)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=0, collate_fn=collate_wrapper)  # , prefetch_factor=3)