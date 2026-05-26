
dataset_dict = {
      'MDTB':'Md',
      'Nishimoto':'Ni',
      'HCPur100':'Hc',
      'IBC':'Ib',
      'Sompatotopic':'So',
      'Social':'Sc',
      'Language':'La'}


def train_group_model(config, mname, save_path=None):
   """
   training a specific model based on the config file created
   model will be trained on cerebellar voxels and average within cortical tessels.
   Args:
      config (dict)      - dictionary with configuration parameters
   Returns:
      conn_model_list (list)    - list of trained models on the list of subjects / log-alphas
      config (dict)             - dictionary containing info for training. Can be saved as json
      train_df (pd.DataFrame)   - dataframe containing training information
   """
   # get dataset class'
   dataset = fdata.get_dataset_class(gl.base_dir,
                                    dataset=config["train_dataset"])

   ## loop over sessions chosen through train_id and concatenate data
   info_list = []

   # Generate model name and create directory
   if mname is None:
      tname = [dataset_dict[d] for d in config['train_dataset']].join('')
      # Join list of strings into string
      mname = f"group_{tname}_{config['parcellation']}_{config['method']}"
   if save_path is None:
      save_path = os.path.join(gl.conn_dir,config['cerebellum'],'train',mname)
   try:
      os.makedirs(save_path)
   except OSError:
      pass

   # Check if training file already exists:
   train_info_name = save_path + "/" + mname + ".tsv"
   if os.path.isfile(train_info_name) and config["append"]:
      train_info = pd.read_csv(train_info_name, sep="\t")
   else:
      train_info = pd.DataFrame()

   # Loop over datasets
   XXX = []
   YYY = []

   for i, ds in enumerate(config["train_dataset"]):
      YY, info, _ = fdata.get_dataset(gl.base_dir,
                                    ds,
                                    atlas=config["cerebellum"],
                                    sess='all',
                                    type='CondHalf',
                                    subj='group')
      XX, info, _ = fdata.get_dataset(gl.base_dir,
                                    ds,
                                    atlas=config["cortex"],
                                    sess='all',
                                    type=config["type"],
                                    subj='group')
      # Average the cortical data over pacels
      X_atlas, _ = at.get_atlas(config['cortex'],gl.atlas_dir)
      # get the vector containing tessel labels
      X_atlas.get_parcel(config['label_img'], unite_struct = False)
      # get the mean across tessels for cortical data
      XX, labels = fdata.agg_parcels(XX, X_atlas.label_vector,fcn=np.nanmean)

      # Remove Nans
      Y = np.nan_to_num(YY[0,:,:])
      X = np.nan_to_num(XX[0,:,:])

      # Add rest condition?
      if config["add_rest"]:
         Y,_ = add_rest(Y,info)
         X,info = add_rest(X,info)

      # train only on some runs?
      if config["train_run"]!='all':
         if isinstance(config["train_run"], list):
            run_mask = info['run'].isin(config["train_run"])
            Y = Y[run_mask.values, :]
            X = X[run_mask.values, :]
            info = info[run_mask]

      # train only on some conds?
      if config['train_cond_num']!='all':
         if isinstance(config["train_cond_num"], list):
            cond_mask = info['cond_num'].isin(config["train_cond_num"])
            Y = Y[cond_mask.values, :]
            X = X[cond_mask.values, :]
            info = info[cond_mask]

      #Definitely subtract intercept across all conditions
      X = (X - X.mean(axis=0))
      Y = (Y - Y.mean(axis=0))

      if 'std_cortex' in config.keys():
         X = std_data(X,config['std_cortex'])
      if 'std_cerebellum' in config.keys():
         Y = std_data(Y,config['std_cerebellum'])

      # cross the halves within each session
      if config["crossed"] is not None:
         Y = cross_data(Y,info,config["crossed"])

      for la in config["logalpha"]:
      # loop over subjects and train models
         print(f'- Train {sub} {config["method"]} logalpha {la}')

         if la is not None:
            # Generate new model
            alpha = np.exp(la) # get alpha
            conn_model = getattr(model, config["method"])(alpha)
            mname_spec = f"{mname}_A{la}_{sub}"
         else:
            conn_model = getattr(model, config["method"])()
            mname_spec = f"{mname}_{sub}"

         # Fit model, get train and validate metrics
         if config["method"] == 'L2reg':
            conn_model.fit(X, Y, info)
         elif config["method"] == 'L2reghalf':
            conn_model.fit(X, Y, config, info)
         elif config["method"] == 'L2reg2':
            conn_model.fit(X, Y, info)
         else:
            conn_model.fit(X, Y)
         R_train,R2_train = train_metrics(conn_model, X, Y)
         # conn_model_list.append(conn_model)

         # collect train metrics ( R)
         model_info = {
                        "subj_id": sub,
                        "mname": mname_spec,
                        "R_train": R_train,
                        "R2_train": R2_train,
                        "num_regions": X.shape[1],
                        "logalpha": la
                        }

         # run cross validation and collect metrics (rmse and R)
         if config['validate_model']:
            R_cv = validate_metrics(conn_model, X, Y, config["cv_fold"][0])
            model_info.update({"R_cv": conn_model.R_cv})

         # Copy over all scalars or strings from config to eval dict:
         for key, value in config.items():
            if not isinstance(value, (list, dict,pd.Series,np.ndarray)):
               model_info.update({key: value})
         # Save the individuals info files
         cio.save_model(conn_model,model_info,save_path + "/" + mname_spec)
         train_info = pd.concat([train_info,pd.DataFrame(model_info)],ignore_index= True)
   train_info.to_csv(train_info_name,sep='\t')
   return config, conn_model_list, train_info
