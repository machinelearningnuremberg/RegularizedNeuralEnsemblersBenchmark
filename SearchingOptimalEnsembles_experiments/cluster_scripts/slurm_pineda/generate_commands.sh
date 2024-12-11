for project in RQ6_qt_micro_nll RQ6_qt_mini_nll RQ6_nb_micro_nll RQ6_nb_mini_nll RQ6_ftc_extended_nll RQ6_tr_version3class_nll RQ6_tr_version3reg_mse
do
    python generate_commands.py --project_name ${project}
done