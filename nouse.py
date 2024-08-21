elif args.task == 'kica_staging':
    args.n_classes=2
    dataset_params['csv_path'] = 'dataset_csv/kica_staging_npy.csv'
    dataset_params['label_dict'] = {'late':0, 'early': 1}
    dataset_params['data_mag'] = '10x512'
    if args.model_type in ['nicwss', 'nic']:
        dataset = NIC_MIL_Dataset(**dataset_params)
    else:
        dataset = Generic_MIL_Dataset(**dataset_params)