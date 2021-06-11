
num_boost_round =999

columns1=['main_price', 'MAIN_SIRE_NAME', 'code_desc', 'CED EPD', 'BW EPD',
       'WW EPD', 'YW EPD', 'RADG EPD', 'DMI EPD', 'YH EPD', 'SC EPD',
       'Doc EPD', 'HP EPD', 'CEM EPD', 'Milk EPD', 'MW EPD', 'MH EPD', '$EN',
       'CW EPD', 'Marb EPD', 'RE EPD', 'Fat EPD', 'FOOT_ANGLE_EPD',
       'FOOT_CLAW_SET_EPD', 'pap_epd', '$W', '$M', '$F', '$G', '$B', '$C',
       '$AxH', '$AxJ', 'Original_Animal_Order']

columns2=['main_price',  'CED EPD', 'BW EPD',
       'WW EPD', 'YW EPD', 'RADG EPD', 'DMI EPD', 'YH EPD', 'SC EPD',
       'Doc EPD', 'HP EPD', 'CEM EPD', 'Milk EPD', 'MW EPD', 'MH EPD', '$EN',
       'CW EPD', 'Marb EPD', 'RE EPD', 'Fat EPD', 'FOOT_ANGLE_EPD',
       'FOOT_CLAW_SET_EPD', 'pap_epd', '$W', '$M', '$F', '$G', '$B', '$C',
       '$AxH', '$AxJ']

columns3=['main_price', 'CED_EPD', 'BW_EPD', 'WW_EPD', 'YW_EPD', 'RADG_EPD',
                                'DMI_EPD', 'YH_EPD', 'SC_EPD', 'Doc_EPD', 'HP_EPD', 'CEM_EPD',
                                'Milk_EPD', 'MW_EPD', 'MH_EPD', 'EN', 'CW_EPD', 'Marb_EPD', 'RE_EPD',
                                'Fat_EPD', 'FOOT_ANGLE_EPD', 'FOOT_CLAW_SET_EPD', 'pap_epd', 'W', 'M',
                                'F', 'G', 'B', 'C', 'AxH', 'AxJ']

# features for training and prediction
features=['CED_EPD', 'BW_EPD', 'WW_EPD', 'YW_EPD', 'RADG_EPD',
                                'DMI_EPD', 'YH_EPD', 'SC_EPD', 'Doc_EPD', 'HP_EPD', 'CEM_EPD',
                                'Milk_EPD', 'MW_EPD', 'MH_EPD', 'EN', 'CW_EPD', 'Marb_EPD', 'RE_EPD',
                                'Fat_EPD', 'FOOT_ANGLE_EPD', 'FOOT_CLAW_SET_EPD', 'pap_epd', 'W', 'M',
                                'F', 'G', 'B', 'C', 'AxH', 'AxJ']
## specify phi relevance values, Smogn Balancing
rg_mtrx = [

    [29000,  0, 0],  ## under-sample
    [35000, 0, 0],  ## under-sample ("majority")
    [50000, 0, 0], ##over-sample ("minority")
    [300000, 1, 0],  ## over-sample ("minority")
    [400000, 1, 0],  ## over-sample ("minority")
    [500000, 1, 0],  ## over-sample ("minority")
    [700000, 1, 0],  ## over-sample ("minority")
    [800000, 1, 0],  ## over-sample ("minority")
    [900000, 1, 0],  ## over-sample ("minority")
    [1500000, 1, 0],  ## over-sample ("minority")
]

params_xg ={'max_depth': (3, 20),
             'gamma': (0, 1),
             'colsample_bytree': (0.3, 0.9),
             'learning_rate' : (0.01,0.3),
             'n_estimators': (50,500),
             'eta':(.005,.5),
             'min_child_weight': (2,8)}