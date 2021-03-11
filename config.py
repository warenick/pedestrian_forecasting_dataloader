import numpy as np
SDD_scales = {
    "bookstore_0": {
        "scale": 0.038392063,
        "certainty": 1.0,
    },
    "bookstore_1": {
        "scale": 0.039892913,
        "certainty": 1.0,
    },
    "bookstore_2": {
        'scale': 0.04062433,
        'certainty': 1.0,
    },
    "bookstore_3": {
        "scale": 0.039098596,
        "certainty": 1.0,
    },
    "bookstore_4": {
        "scale": 0.0396,
        "certainty": 1.0},

    "bookstore_5": {
        "scale": 0.0396,
        "certainty": 0.9},

    "bookstore_6": {
        "scale": 0.0413,
        "certainty": 0.9,
    },

    "coupa_0": {
        "scale": 0.027995674,
        "certainty": 0.9,
    },

    "coupa_1": {
        "scale": 0.023224545,
        "certainty": 0.9,
    },

    "coupa_2": {
        "scale": 0.024,
        "certainty": 0.9,
    },

    "coupa_3": {
        "scale": 0.025,
        "certainty": 0.9,
    },

    "deathCircle_0": {
        "scale": 0.04064,
        "certainty": 0.9,
    },

    "deathCircle_1": {
        "scale": 0.039076923,
        "certainty": 0.9,
    },

    "deathCircle_2": {
        "scale": 0.03948382,
        "certainty": 0.9,
    },

    "deathCircle_3": {
        "scale": 0.028478209,
        "certainty": 0.9,
    },

    "deathCircle_4": {
        "scale": 0.038980137,
        "certainty": 0.9,
    },

    "gates_0": {
        "scale": 0.038980137,
        "certainty": 0.9,
    },
    "gates_1": {
        "scale": 0.03770837,
        "certainty": 0.9,
    },
    "gates_2": {
        "scale": 0.037272793,
        "certainty": 0.9,
    },
    "gates_3": {
        "scale": 0.034515323,
        "certainty": 0.9,
    },
    "gates_4": {
        "scale": 0.04412268,
        "certainty": 0.9,
    },
    "gates_5": {
        "scale": 0.0342392,
        "certainty": 0.9,
    },
    "gates_6": {
        "scale": 0.0342392,
        "certainty": 0.9,
    },
    "gates_7": {
        "scale": 0.04540353,
        "certainty": 0.9,
    },
    "gates_8": {
        "scale": 0.045191525,
        "certainty": 0.9,
    },
    "hyang_0": {
        "scale": 0.034749693,
        "certainty": 0.9,
    },
    "hyang_1": {
        "scale": 0.0453136,
        "certainty": 0.9,
    },
    "hyang_2": {
        "scale": 0.054992233,
        "certainty": 0.9,
    },
    "hyang_3": {
        "scale": 0.056642,
        "certainty": 0.9,
    },
    "hyang_4": {
        "scale": 0.034265612,
        "certainty": 0.9,
    },
    "hyang_5": {
        "scale": 0.029655497,
        "certainty": 0.9,
    },
    "hyang_6": {
        "scale": 0.052936449,
        "certainty": 0.9,
    },
    "hyang_7": {
        "scale": 0.03540125,
        "certainty": 0.9,
    },
    "hyang_8": {
        "scale": 0.034592381,
        "certainty": 0.9,
    },
    "hyang_9": {
        "scale": 0.038031423,
        "certainty": 0.9,
    },
    "hyang_10": {
        "scale": 0.054460944,
        "certainty": 0.9,
    },
    "hyang_11": {
        "scale": 0.054992233,
        "certainty": 0.9,
    },
    "hyang_12": {
        "scale": 0.054104065,
        "certainty": 0.9,
    },
    "hyang_13": {
        "scale": 0.0541,
        "certainty": 0.9,
    },
    "hyang_14": {
        "scale": 0.0541,
        "certainty": 0.9,
    },

    "nexus_0": {
            "scale": 0.043986494,
            "certainty": 0.9,
        },

    "nexus_1": {
        "scale": 0.043316805,
        "certainty": 0.9,
    },

    "nexus_2": {
        "scale": 0.042247434,
        "certainty": 0.9,
    },

    "nexus_3": {
        "scale": 0.045883871,
        "certainty": 0.9,
    },

    "nexus_4": {
        "scale": 0.045883871,
        "certainty": 0.9,
    },

    "nexus_5": {
        "scale": 0.045395745,
        "certainty": 0.9,
    },

    "nexus_6": {
        "scale": 0.037929168,
        "certainty": 0.9,
    },

    "nexus_7": {
        "scale": 0.037106087,
        "certainty": 0.9,
    },

    "nexus_8": {
        "scale": 0.037106087,
        "certainty": 0.9,
    },

    "nexus_9": {
        "scale": 0.044917895,
        "certainty": 0.9,
    },

    "nexus_10": {
        "scale": 0.043991753,
        "certainty": 0.9,
    },

    "nexus_11": {
        "scale": 0.043766154,
        "certainty": 0.9,
    },



}


students_pix_to_image_cfg = {
    "displ_y": (576)//2,
    "displ_x": (720)//2,
    "coef_x": 1,
    "coef_y": -1
}

zara2_pix_to_image_cfg = {
    "displ_y": (576+22)//2,
    "displ_x": (720)//2,
    "coef_x": 1,
    "coef_y": -1
}
zara_cfg = {"scale": np.array([[-2.595651699999999840e-02, -5.157280400000000145e-18, 7.838868099999996453e+00],
                                [-1.095387399999999886e-03, 2.166433000000000247e-02, 5.566045600000004256e+00],
                                [1.954012500000000506e-20, 4.217141000000002596e-19, 1.000000000000000444e+00]])
            }  # pix to meters,   x axis inverted? starting from [?, ?]

student_cfg = {"scale": np.array([[0.02104651,  0.,         7.57676355],
                                  [0.,         0.02386598, 6.87340224],
                                  [0.,         0.,         1.]])
               }  # pix to meters! pixels from [-359, -288] to [359, 288] !!!


cropping_cfg = {
    "agent_center": [0.25, 0.5],
    "image_area_meters": [15, 15],
    "image_shape": [192, 192],
}

cfg = {
    "data_path": "../data/train/",
    "files": ["biwi/biwi_hotel.txt", "mot/PETS09-S2L1.txt"],
    'raster_params': {
        "draw_hist": True,
        "use_map": False,
        "normalize": False,
        # 'raster_size': [224, 224],
        # 'pixel_size': [0.5, 0.5],
        # 'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    'model_params': {
        'history_num_frames': 8,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 12,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "vis_attention",
        'lr': 1e-4,
        'weight_path': None,
        'train': 1,
        'predict': 0,
    },
    "SDD_scales": SDD_scales,
    "zara_h": zara_cfg,
    "student_h": student_cfg,
    "cropping_cfg": cropping_cfg,
    "students_pix_to_image_cfg": students_pix_to_image_cfg,
    "zara2_pix_to_image_cfg": zara2_pix_to_image_cfg,

}