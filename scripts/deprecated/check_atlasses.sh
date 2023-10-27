


atlas_dir=/Users/callithrix/code/Python/cerebellar_atlases/Nettekoven_2023
Ks=('32' '68' '128')
symmetries=('Sym' 'Asym')
for k in ${Ks}; do
    for sym in ${symmetries}; do

        echo "Displaying ${sym} ${k} atlas"

        fsleyes \
        /Users/callithrix/Documents/Projects/Functional_Fusion/Atlases/tpl-MNI152NLin6AsymC/tpl-MNI152NLin6AsymC_T1w.nii \
        ${atlas_dir}/atl-Nettekoven${sym}${k}_space-MNI152NLin6AsymC_dseg.nii \
        ${atlas_dir}/atl-Nettekoven${sym}${k}_space-MNI152NLin6AsymC_probseg.nii.gz -cm red-yellow &

        fsleyes \
        /Users/callithrix/Documents/Projects/Functional_Fusion/Atlases/tpl-MNI152NLin2009cSymC/tpl-MNI152NLin2009cSymC_T1w.nii \
        ${atlas_dir}/atl-Nettekoven${sym}${k}_space-MNI152NLin2009cSymC_dseg.nii \
        ${atlas_dir}/atl-Nettekoven${sym}${k}_space-MNI152NLin2009cSymC_probseg.nii.gz -cm red-yellow &

        fsleyes \
        /Applications/spm12/toolbox/suit/atlasesSUIT/SUIT.nii \
        ${atlas_dir}/atl-Nettekoven${sym}${k}_space-SUIT_dseg.nii \
        ${atlas_dir}/atl-Nettekoven${sym}${k}_space-SUIT_probseg.nii.gz -cm red-yellow
    
    done
done

