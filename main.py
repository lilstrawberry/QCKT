import pandas as pd
import torch
import torch.nn as nn

from run import run_epoch
from model import DKT_QCKT
import glo

mp2path = {
    'static11': {
        'ques_skill_path': 'data/Statics2011/ques_skill.csv',
        'train_path': 'data/Statics2011/train_question.txt',
        'test_path': 'data/Statics2011/test_question.txt',
        'pre_load_gcn': 'data/Statics2011/static_ques_skill_gcn_adj.pt',
        'positive_matrix_path': 'data/Statics2011/Static11_Q_Q_sparse.pt',
        'unique_positive_matrix_path': 'data/Statics2011/static11_unique_skill_Q-Q',
        'skill_max': 106,
        'epoch': 200
    }
}

use_dataset = ['static11']

if __name__ == '__main__':

    glo._init()

    for dataset in use_dataset:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ques_skill_path = mp2path[dataset]['ques_skill_path']

        train_path = mp2path[dataset]['train_path']
        if "valid_path" in mp2path[dataset]:
            valid_path = mp2path[dataset]['valid_path']
        else:
            valid_path = mp2path[dataset]['test_path']
        test_path = mp2path[dataset]['test_path']

        pro_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 0])
        skill_max = mp2path[dataset]['skill_max']

        gcn_matrix = torch.load(mp2path[dataset]['pre_load_gcn']).to(device)

        pos_matrix = torch.load(mp2path[dataset]['positive_matrix_path']).to(device)
        unique_pos_matrix = torch.load(mp2path[dataset]['unique_positive_matrix_path']).to(device)

        pro2skill = torch.zeros((pro_max, skill_max)).to(device)
        csv_data = pd.read_csv(ques_skill_path).values

        for x, y in zip(csv_data[:, 0], csv_data[:, 1]):
            pro2skill[x][y] = 1

        glo.set_value('gcn_matrix', gcn_matrix)
        glo.set_value('pro2skill', pro2skill)
        glo.set_value('unique_pos_matrix', unique_pos_matrix)
        glo.set_value('pos_matrix', pos_matrix)

        p = 0.4
        phi = 0.01
        d = 128
        learning_rate = 0.002
        epochs = 70
        batch_size = 80
        min_seq = 3
        max_seq = 200
        grad_clip = 15.0
        patience = 15

        avg_auc = 0
        avg_acc = 0

        sublist = []

        for now_step in range(5):

            best_acc = 0
            best_auc = 0
            state = {'auc': 0, 'acc': 0, 'loss': 0}

            model = DKT_QCKT(pro_max, skill_max, d, p, phi)
            model = model.to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

            one_p = 0

            for epoch in range(mp2path[dataset]['epoch']):

                one_p += 1

                train_loss, train_acc, train_auc = run_epoch(pro_max, train_path, batch_size,
                                                             True, min_seq, max_seq, model, optimizer, criterion,
                                                             device,
                                                             grad_clip)
                print(
                    f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}')

                valid_loss, valid_acc, valid_auc = run_epoch(pro_max, valid_path, batch_size, False,
                                                          min_seq, max_seq, model, optimizer, criterion, device,
                                                          grad_clip)

                print(f'epoch: {epoch}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, valid_auc: {valid_auc:.4f}')

                sublist.append(valid_auc)

                if valid_auc > best_auc:
                    one_p = 0
                    best_auc = valid_auc
                    best_acc = valid_acc
                    torch.save(model.state_dict(), f"./DKT_QCKT_{dataset}_{now_step}_model.pkl")
                    torch.save(state, f'./DKT_QCKT_{dataset}_{now_step}_state.ckpt')

                    if one_p >= patience:
                        break

            model.load_state_dict(torch.load(f'./DKT_QCKT_{dataset}_{now_step}_model.pkl'))
            test_loss, test_acc, test_auc = run_epoch(pro_max, test_path, batch_size, False,
                                                         min_seq, max_seq, model, optimizer, criterion, device,
                                                         grad_clip)

            state['auc'] = test_auc
            state['acc'] = test_acc
            state['loss'] = test_loss

            print(f'*******************************************************************************')
            print(f'test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}')
            print(f'*******************************************************************************')

            avg_auc += test_auc
            avg_acc += test_acc

        avg_auc = avg_auc / 5
        avg_acc = avg_acc / 5
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')