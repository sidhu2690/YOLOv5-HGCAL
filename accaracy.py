import numpy as np
import glob
import os
import ROOT as rt

a = glob.glob('../datasets/muon47_gPU_n/labels/test/Event*.txt')
print(len(a))
n_true = 0
n_predict = 0
h_eta_predict = rt.TH1F('h_eta_predict', '', 200, -1, 1)
h_eta_abspredict = rt.TH1F('h_eta_abspredict', '', 200, 0, 0.2)
h_phi_predict = rt.TH1F('h_phi_predict', '', 200, -1, 1)
h_phi_abspredict = rt.TH1F('h_phi_abspredict', '', 200, 0, 0.2)
h_dis_predict = rt.TH1F('h_dis_predict', '', 200, 0, 0.2)
h_pro_predict = rt.TH1F('h_pro_predict', 'Propability of predection', 100, 0., 1.)
h_double_dis = rt.TH1F('h_double_dis', 'dr_2prec', 100, 0., 1.)
f_mis = open('missclassified_n_2.txt', 'w')
for i in range(len(a)):
    lable_true = np.loadtxt(a[i])
    file_name = os.path.basename(a[i])
    #print(file_name)
    if os.path.exists('runs/detect/exp53/labels/' + file_name):
        lable_predict = np.loadtxt('runs/detect/exp53/labels/' + file_name)
    else:
        lable_predict = np.array([])
    if np.size(lable_predict) > 6:
        print(lable_predict)
        amax = np.argmax(lable_predict[:, 5])
        if np.shape(lable_predict)[0] == 2:
            print(lable_predict[0,5], '   ', lable_predict[1,5])
            h_double_dis.Fill(np.sqrt(((lable_predict[0,2] - lable_predict[1,2])*2.3)**2 + ((lable_predict[0,2] - lable_predict[0,2])*6.28)**2))
        lable_predict = lable_predict[amax, :]
    lable_predict = lable_predict.flatten()
    n_true += np.size(lable_true)/5
    n_predict += np.size(lable_predict)/6
    if np.size(lable_predict) !=0:
        h_pro_predict.Fill(lable_predict[5])
    if np.size(lable_true) != 0 and np.size(lable_predict) != 0:
        distance = np.sqrt(((lable_true[1] - lable_predict[1])*2.3)**2 + ((lable_true[2] - lable_predict[2])*6.28)**2)
        h_eta_predict.Fill((lable_true[2] - lable_predict[2])*2.3)
        h_eta_abspredict.Fill(np.abs(lable_true[2] - lable_predict[2]) * 2.3)
        h_phi_predict.Fill((lable_true[1] - lable_predict[1])*6.28)
        h_phi_abspredict.Fill(np.abs(lable_true[1] - lable_predict[1]) *6.28)
        h_dis_predict.Fill(distance)
    else:
        distance = 0
    if np.size(lable_true)/5 != np.size(lable_predict)/6 or distance > 0.12:
        print('error: not detected correctly')
        print(file_name)
        eve_num = ''
        #print([num for num in file_name.split() if num.isdigit()])
        eve_num = eve_num.join([num for num in file_name if num.isdigit()])
        print(eve_num)
        f_mis.write(eve_num + '\n')
f_mis.close()
outfile = rt.TFile('predict_hist_pun_2.root', 'RECREATE')
outfile.cd()
h_eta_predict.Write()
h_eta_abspredict.Write()
h_phi_predict.Write()
h_phi_abspredict.Write()
h_dis_predict.Write()
h_pro_predict.Write()
h_double_dis.Write()
outfile.Close()
print(n_true, "   ", n_predict)
