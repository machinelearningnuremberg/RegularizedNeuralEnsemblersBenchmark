# ./per_project_run.sh RQ1_nb_mini_nll 3 greedy1 2
# ./per_project_run.sh RQ1_nb_mini_error 3 greedy1 2

# for baseline in akaike1r1 akaike2r1
# do
#     ./per_project_run.sh RQ2_qt_micro_nll 30 ${baseline} 2
#     ./per_project_run.sh RQ2_qt_mini_nll 30 ${baseline} 2
#     ./per_project_run.sh RQ2_nb_micro_nll 3 ${baseline} 2
#     ./per_project_run.sh RQ2_nb_mini_nll 3 ${baseline} 2
#     ./per_project_run.sh RQ2_ftc_extended_nll 6 ${baseline} 2
#     ./per_project_run.sh RQ2_tr_version3class_nll 83 ${baseline} 2
#     ./per_project_run.sh RQ2_tr_version3reg_mse 17 ${baseline} 2
# done

# for baseline in sks1r1 sks2r1 
# do
#     ./per_project_run.sh RQ2_qt_micro_nll 30 ${baseline} 3
#     ./per_project_run.sh RQ2_qt_mini_nll 30 ${baseline} 3
#     ./per_project_run.sh RQ2_nb_micro_nll 3 ${baseline} 3
#     ./per_project_run.sh RQ2_nb_mini_nll 3 ${baseline} 3
#     ./per_project_run.sh RQ2_ftc_extended_nll 6 ${baseline} 3
#     ./per_project_run.sh RQ2_tr_version3class_nll 83 ${baseline} 3
#     ./per_project_run.sh RQ2_tr_version3reg_mse 17 ${baseline} 3
# done

for baseline in neural1r1
do
    ./per_project_run.sh RQ5_qt_micro_nll 30 ${baseline} 5
    ./per_project_run.sh RQ5_qt_mini_nll 30 ${baseline} 5
    ./per_project_run.sh RQ5_nb_micro_nll 3 ${baseline} 5
    ./per_project_run.sh RQ5_nb_mini_nll 3 ${baseline} 5
    ./per_project_run.sh RQ5_ftc_extended_nll 6 ${baseline} 5
    ./per_project_run.sh RQ5_tr_version3class_nll 83 ${baseline} 5
    ./per_project_run.sh RQ5_tr_version3reg_mse 17 ${baseline} 5
done

# baseline=neural1r1
# # ./per_project_run.sh RQ6_qt_micro_nll 30 ${baseline} 6
# # ./per_project_run.sh RQ6_qt_mini_nll 30 ${baseline} 6
# # ./per_project_run.sh RQ6_nb_micro_nll 3 ${baseline} 6
# # ./per_project_run.sh RQ6_nb_mini_nll 3 ${baseline} 6
# # ./per_project_run.sh RQ6_ftc_extended_nll 6 ${baseline} 6
# # ./per_project_run.sh RQ6_tr_version3class_nll 83 ${baseline} 6
# ./per_project_run.sh RQ6_tr_version3reg_mse 17 ${baseline} 6