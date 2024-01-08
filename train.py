from torch_geometric.loader import DataLoader
from dataset import MyDataset
import glob
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
from models.model_one import vit_gcn, gcn, vit
from utils.GCN_utils import *
import json
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--train_npz_dir', type=str, default='data/train.csv', help='training set dir')
parser.add_argument('--val_npz_dir', type=str, default='data/test.csv', help='validation set dir')
parser.add_argument('--test_npz_dir', type=str, default='data/test', help='test set dir')
parser.add_argument('--log_dir', type=str, default='log/', help='validation set dir')
parser.add_argument('--model_save_dir', type=str, default='model_save/', help='model save dir')
parser.add_argument('--seed', type=int, default=1024, help='Random seed.')
parser.add_argument('--fold', type=int, default=0, help='fold number')
parser.add_argument('--b', type=int, default=4096, choices=[256, 4096], help='patch size')
parser.add_argument('--batch_size', type=int, default=8, help='mini_batch size')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--save', type=bool, default=True,
                    help='.')
parser.add_argument('--tensorboard', type=bool, default=True,
                    help='.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--loss_weight', type=float, default=0.7, help='loss_weight alpha.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
args = parser.parse_args()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False


def train(epoch, save=True, tsboard=True):
    global losss
    model.train()
    scheduler.step()
    epoch_loss, epoch_loss1, epoch_loss2 = 0, 0, 0
    # alpha = args.loss_weight
    id, y, status, risk, pred, label = [], [], [], [], [], []

    for ii, data in tqdm(enumerate(train_batch_data)):
        # 清空梯度
        optimizer.zero_grad()

        # 读取数据
        data = data.to(device)
        t_status = data.status
        t_time = data.time
        t_label = data.label
        t_patient_id = data.id
        t_y = t_time

        # 单分支输出结果值，预测结果为 组织分型
        t_pred = model(data)
        # 双分支输出结果值，预测结果为 风险值+组织分型
        # t_risk, t_pred = model(data)

        # 单分支的loss
        loss = loss_func(t_pred, t_label)
        # 双分支的loss
        # loss1 = cox_loss(t_risk, t_time, t_status) * alpha
        # loss2 = loss_func(t_pred, t_label) * (1 - alpha)

        # 单分支训练集loss
        train_loss = loss
        # 双分支训练集loss
        # train_loss = loss1 + loss2

        train_loss.backward()
        optimizer.step()

        epoch_loss += train_loss.item()
        # 双分支
        # epoch_loss1 += loss1.item()
        # epoch_loss2 += loss2.item()

        pred.extend(t_pred.tolist())
        label.extend(t_label.tolist())
        # 计算risk时加上
        # id.extend(t_patient_id.tolist())
        # risk.extend(t_risk.tolist())
        # status.extend(t_status.tolist())
        # y.extend(t_y.tolist())

    probs = np.array(pred)
    # 将类别索引转换为单热编码形式
    pred = np.argmax(probs, axis=1)
    label = np.argmax(label, axis=1)
    # 计算risk时加上
    # id = np.array(id)
    # y = np.array(y)
    # status = np.array(status)
    # risk = np.array(risk)

    # 计算生长类型指标
    accuracy = accuracy_score(label, pred)
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')

    # 计算c_index和hr
    # patient_list, uni_idx = np.unique(id, return_index=True)
    # num_patients = len(patient_list)
    # counter_matrix = np.zeros(num_patients)
    # risk_matrix = np.zeros(num_patients)
    # label_matrix = y[uni_idx]
    # status_matrix = status[uni_idx]
    #
    # for ii, patient in enumerate(id):
    #     risk_matrix[np.argwhere(patient_list == patient)] += risk[ii]
    #     counter_matrix[np.argwhere(patient_list == patient)] += 1
    # risk_matrix = risk_matrix / counter_matrix
    #
    # c_index, haztio_ratio = calu_cindex_hr(label_matrix, risk_matrix, status_matrix)

    if save:
        torch.save(model, args.model_save_dir + 'epoch%d.pth' % epoch)
    if tsboard:
        writer.add_scalars("train_loss", {'Train': epoch_loss / len(train_data)}, e)
    print('==========epoch%s===========' % epoch)
    print('train loss:%f, Accuracy:%.4f, Precision:%.4f, Recall:%.4f, F1-Score:%.4f'
          % (epoch_loss / len(train_data), accuracy, precision, recall, f1))
    # 多分支
    # print('train loss:%f, loss1:%f, loss2:%f, c-index:%.4f, HR:%.4f, Accuracy:%.4f, Precision:%.4f, Recall:%.4f, '
    #       'F1-Score:%.4f' % (epoch_loss / len(train_data), epoch_loss1 / len(train_data), epoch_loss2 / len(train_data),
    #                          c_index, haztio_ratio, accuracy, precision, recall, f1))

    # validation
    model.eval()
    val_epoch_loss = 0
    id, y, status, risk, pred, label = [], [], [], [], [], []

    for jj, v_data in enumerate(val_batch_data):
        # 读取数据
        v_data = v_data.to(device)
        v_time = v_data.time
        v_status = v_data.status
        v_label = v_data.label
        v_patient_id = v_data.id
        v_y = v_time

        with torch.no_grad():
            # 单分支输出结果值，预测结果为 组织分型
            v_pred = model(v_data)
            # 双分支输出结果值，预测结果为 风险值+组织分型
            # v_risk, v_pred = model(v_data)

        # 单分支的loss
        v_loss = loss_func(v_pred, v_label)
        # 双分支的loss
        # v_loss1 = cox_loss(v_risk, v_time, v_status) * alpha
        # v_loss2 = loss_func(v_pred, v_label) * (1 - alpha)

        # 单分支测试集loss
        val_loss = v_loss
        # 双分支测试集loss
        # val_loss = v_loss1 + v_loss2

        val_epoch_loss += val_loss.item()

        pred.extend(v_pred.tolist())
        label.extend(v_label.tolist())
        # 计算risk时加上
        # y.extend(v_y.tolist())
        # status.extend(v_status.tolist())
        # risk.extend(v_risk.tolist())
        # id.extend(v_patient_id.tolist())

    probs = np.array(pred)
    # 将类别索引转换为单热编码形式
    pred = np.argmax(probs, axis=1)
    label = np.argmax(label, axis=1)
    # 计算risk时加上
    # id = np.array(id)
    # y = np.array(y)
    # status = np.array(status)
    # risk = np.array(risk)

    # 计算生长类型的指标
    accuracy = accuracy_score(label, pred)
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')

    # 计算c_index和hr
    # patient_list, uni_idx = np.unique(id, return_index=True)
    # num_patients = len(patient_list)
    # counter_matrix = np.zeros(num_patients)
    # risk_matrix = np.zeros(num_patients)
    # label_matrix = y[uni_idx]
    # status_matrix = status[uni_idx]
    #
    # for ii, patient in enumerate(id):
    #     risk_matrix[np.argwhere(patient_list == patient)] += risk[ii]
    #     counter_matrix[np.argwhere(patient_list == patient)] += 1
    # risk_matrix = risk_matrix / counter_matrix
    #
    # c_index, haztio_ratio = calu_cindex_hr(label_matrix, risk_matrix, status_matrix)

    print('val loss: %f, Accuracy:%.4f, Precision:%.4f, Recall:%.4f, F1-Score:%.4f' %
          (val_epoch_loss / len(val_data), accuracy, precision, recall, f1))
    # 双分支
    # print('val loss: %f, c-index:%.4f, HR:%.4f, Accuracy:%.4f, Precision:%.4f, Recall:%.4f, F1-Score:%.4f' %
    #       (val_epoch_loss / len(val_data), c_index, haztio_ratio, accuracy, precision, recall, f1))

    if losss > val_epoch_loss / len(val_data):
        losss = val_epoch_loss / len(val_data)
        if save:
            print('model saved!')
            model_save_fold = os.path.join(args.model_save_dir, str(b), str(args.fold), 'best_validation.pth')
            torch.save(model, model_save_fold)
    if tsboard:
        writer.add_scalar("val_Acc", accuracy, e)
        writer.add_scalar("val_Pre", precision, e)
        writer.add_scalar("val_Rec", recall, e)
        writer.add_scalar("val_F1", f1, e)
        # 双分支
        # writer.add_scalar("val_c-index", c_index, e)
        # writer.add_scalar("val_HR", haztio_ratio, e)
        writer.add_scalars("val_loss", {'Validation': val_epoch_loss / len(val_data)}, e)
    make_dirs(args.log_dir)
    with open(os.path.join(args.log_dir, str(b), str(args.fold), 'BLCA_freeze_risk_nocost_val.json'), 'a') as j:
        json.dump(
            (str(epoch), str(val_epoch_loss / len(val_data)), str(accuracy), str(precision), str(recall), str(f1)), j)
        # 双分支
        # json.dump(
        #     (str(epoch), str(val_epoch_loss / len(val_data)), str(c_index), str(haztio_ratio), str(accuracy), str(precision), str(recall), str(f1)), j)
        j.write('\n')


# 自动划分五折数据，训练结束后数据归位
def set_data(path, fold, mode='0'):
    if mode == '0':
        assert fold in [0, 1, 2, 3, 4]
        file_pattern = os.path.join(path, "*.npz")
        npz_files = glob.glob(file_pattern)
        file_names = [os.path.basename(file) for file in npz_files]
        fold_num = int(len(file_names) / 5)

        val_data = file_names[fold * fold_num:(fold + 1) * fold_num]
        if fold == 0:
            train_data = file_names[fold_num:]
        elif fold != 4:
            train_data = file_names[:fold * fold_num] + file_names[(fold + 1) * fold_num:]
        elif fold == 4:
            val_data = file_names[-fold_num:]
            train_data = file_names[:-fold_num]
        else:
            train_data = 0
            quit()

        for file_train in train_data:
            shutil.move(os.path.join(path, file_train), os.path.join(path, 'train', file_train))
        for file_val in val_data:
            shutil.move(os.path.join(path, file_val), os.path.join(path, 'val', file_val))

    elif mode == '1':
        data1 = os.listdir(os.path.join(path, 'train'))
        data2 = os.listdir(os.path.join(path, 'val'))
        for item in data1:
            item_path = os.path.join(path, 'train', item)
            shutil.move(item_path, os.path.join(path))
        for item in data2:
            item_path = os.path.join(path, 'val', item)
            shutil.move(item_path, os.path.join(path))


if __name__ == '__main__':
    seed_torch(args.seed)

    tsboard = args.tensorboard
    save = args.save
    b = args.b
    fold = args.fold
    # set_data('data/', fold, mode='0')

    train_data = MyDataset(args.train_npz_dir, b)
    train_batch_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_data = MyDataset(args.val_npz_dir, b)
    val_batch_data = DataLoader(val_data, batch_size=16, shuffle=False)  # batch_size=len(val_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit().to(device)

    # 加载权重
    # risk_dir = os.path.join(args.model_save_dir, str(b), str(args.fold), 'best_validation.pth')
    # paras = torch.load(risk_dir).state_dict()
    # model.load_state_dict(state_dict=paras)

    loss_func = nn.BCELoss()  # loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.5)
    make_dirs(args.model_save_dir)

    if tsboard:
        writer = SummaryWriter()

    losss = 9999
    for e in range(args.epochs):
        train(e, save=save, tsboard=tsboard)

    # set_data('data/', fold, mode='1')
    #
    # risk_freeze_dir = os.path.join(args.model_save_dir, str(b), str(args.fold), 'best_validation.pth')
    # paras_2 = torch.load(risk_freeze_dir).state_dict()
    #
    # model = GCN_Freeze_Risk(in_chan=in_chan).to(device)
    # model.load_state_dict(paras_2)
    #
    # test(require_json='True')
