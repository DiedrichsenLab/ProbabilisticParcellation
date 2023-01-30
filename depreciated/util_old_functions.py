def get_parcel(atlas, parcel_name='MDTB10', do_plot=False):
    """Samples the existing MDTB10 parcellation
    Then displays it as check
    Not sure what the function was used for.... not particularly useful

    """
    atl_dir = base_dir + '/Atlases'
    with open(atl_dir + '/atlas_description.json') as file:
        atlases = json.load(file)
    if atlas not in atlases:
        raise(NameError(f'Unknown Atlas: {atlas}'))
    ainf = atlases[atlas]

    parcel = nb.load(atl_dir + '/%s/atl-%s_space-%s_dseg.nii'
                     % (ainf['dir'], parcel_name, ainf['space']))
    suit_atlas, _ = am.get_atlas(atlas, atl_dir)

    data = suit.reslice.sample_image(parcel,
            suit_atlas.world[0],
            suit_atlas.world[1],
            suit_atlas.world[2],0)

    # Read the parcellation colors: Add additional row for parcel 0
    ########################################################
    # The path of color .lut file to be changed if color info
    # stored in separate atlas folder. Right now, all colors are
    # stored in `tpl-SUIT` folder.
    ########################################################
    color_file = atl_dir + f'/tpl-SUIT/atl-{parcel_name}.lut'
    color_info = pd.read_csv(color_file, sep = ' ',header=None)
    colors = color_info.iloc[:,1:4].to_numpy()

    # Map Plot if requested (for a check)
    if do_plot:
        Nifti = suit_atlas.data_to_nifti(data)
        surf_data = suit.flatmap.vol_to_surf(Nifti,stats='mode')
        fig = suit.flatmap.plot(surf_data,render='plotly',
                                overlay_type='label',cmap=colors)
        fig.show()
    return data, colors

# def write_dlabel_cifti(parcellation, atlas, res='32k'):
#     #TODO: unfinished
#     if res == '32k':
#         VERTICES = 32492
#         bm_name = ['cortex_left', 'cortex_right']
#     else:
#         raise ValueError('Only fs_LR32k template is currently supported!')
#
#     if parcellation.dim() == 1:
#         # reshape to (1, num_vertices)
#         parcellation = parcellation.reshape(1,-1)
#
#     if parcellation.shape[1] == VERTICES*2:
#         # The input parcellation is already the full parcels
#         # (including medial wall)
#         pass
#     else:
#         # If the input parcellation is masked, we restore it
#         # to the full 32k vertices
#         par = np.full((1, VERTICES * 2), 0, dtype=int)
#         if atlas.structure == bm_name:
#             idx = np.hstack((atlas.vertex[0], atlas.vertex[1]+VERTICES))
#         if 'cortex_left' in stru.lower:
#             this_idx = atlas.vertex[i]
#             par[:, this_idx] =

def make_label_cifti(data,
                     anatomical_struct='Cerebellum',
                     labels=None,
                     label_names=None,
                     column_names=None,
                     label_RGBA=None):
    """Generates a label Cifti2Image from a numpy array

    THIS FUNCTION IS NOW PART OF NITOOLS - 

    Args:
        data (np.array):
             num_vert x num_col data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'Cerebellum'
        labels (list): Numerical values in data indicating the labels -
            defaults to np.unique(data)
        label_names (list):
            List of strings for names for labels
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors for labels
    Returns:
        gifti (GiftiImage): Label gifti image

    """
    num_verts, num_cols = data.shape
    if labels is None:
        labels = np.unique(data)
    num_labels = len(labels)

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if label_RGBA is None:
        label_RGBA = np.zeros([num_labels,4])
        hsv = plt.cm.get_cmap('hsv',num_labels)
        color = hsv(np.linspace(0,1,num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        for i in range(num_labels):
            label_RGBA[i] = color[i]

    # Create label names from numerical values
    if label_names is None:
        label_names = []
        for i in labels:
            label_names.append("label-{:02d}".format(i))

    labelDict = [('???',(0,0,0,0))]
    for i, p in enumerate(Data):
        colorValue = (1, 1, 1, 1)
        if (p > 0):
            colorValue = colorMapping_p.to_rgba(p)
        elif (p < 0):
            colorValue = colorMapping_n.to_rgba(p)
        labelDict[p] = (i, colorValue)

    names = ['CIFTI_STRUCTURE_CORTEX_LEFT' for i in range(L_data.shape[0])]
    names.extend(['CIFTI_STRUCTURE_CORTEX_RIGHT' for i in range(R_data.shape[0])])
    verteces = [i for i in range(L_data.shape[0])]
    verteces.extend([i for i in range(L_data.shape[0])])
    verteces = np.asarray(verteces)
    brainModelAxis = nib.cifti2.cifti2_axes.BrainModelAxis(name=names, vertex=np.asarray(verteces),
                                                           nvertices={
                                                               'CIFTI_STRUCTURE_CORTEX_LEFT': 32492,
                                                               'CIFTI_STRUCTURE_CORTEX_RIGHT': 32492}, )
    newLabelAxis = nib.cifti2.cifti2_axes.LabelAxis(['aaa'], labelDict)
    newheader = nib.cifti2.cifti2.Cifti2Header.from_axes((newLabelAxis, brainModelAxis))
    newImage = nib.cifti2.cifti2.Cifti2Image(dataobj=Data.reshape([1, -1]), header=newheader)
    newImage.to_filename('%s/' % dir + name + '.dlabel.nii')


    # Create key-color mapping for labelAxis
    np.apply_along_axis(map(), 0, b)
    d = dict(enumerate(a))
    # Create label.gii structure
    C = nb.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    E_all = []
    for (label, rgba, name) in zip(labels, label_RGBA, label_names):
        E = nb.gifti.gifti.GiftiLabel()
        E.key = label
        E.label= name
        E.red = rgba[0]
        E.green = rgba[1]
        E.blue = rgba[2]
        E.alpha = rgba[3]
        E.rgba = rgba[:]
        E_all.append(E)

    D = list()
    for i in range(num_cols):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL',
            datatype='NIFTI_TYPE_UINT8',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.extend(E_all)
    return gifti