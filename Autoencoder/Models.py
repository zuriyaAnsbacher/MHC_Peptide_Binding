import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Encoder(nn.Module):
    def __init__(self, max_len, encoding_dim, batch_size, model_params=None):
        super().__init__()
        self.max_len = max_len
        self.encoding_dim = encoding_dim
        self.vocab_size = max_len * 20  # + 2
        self.batch_size = batch_size

        if model_params:
            self.embedding_dim = model_params["embedding_dim"]
            self.filter1 = model_params["filter1"]
            self.filter2 = model_params["filter2"]
            self.kernel1 = model_params["kernel1"]
            self.kernel2 = model_params["kernel2"]
            self.dropout_enc = model_params["dropout_enc"]
        else:
            self.embedding_dim = 15
            self.filter1 = 8
            self.filter2 = 16
            self.kernel1 = 2
            self.kernel2 = 3
            self.dropout_enc = 0

        self.size_after_conv = self.calculate_size_after_conv()

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=-1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.filter1, kernel_size=(self.kernel1, self.kernel2), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.filter1, self.filter2, kernel_size=(self.kernel1, self.kernel2), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.size_after_conv, 512),  # 4320
            nn.ReLU(),
            nn.Dropout(self.dropout_enc)
        )
        self.fc2 = nn.Linear(512, self.encoding_dim)

    def forward(self, x):
        x = self.embedding(x.long())
        # input size to Conv2d: (batch, number channels, height, width)
        x = x.view(self.batch_size, self.embedding_dim, -1).unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def calculate_size_after_conv(self):
        x_size = self.embedding_dim
        y_size = self.max_len

        # conv1 -> nn.Conv2d
        x_size = (x_size - self.kernel1) + 1
        y_size = (y_size -self.kernel2) + 1
        # conv1 -> nn.MaxPool2d(2)
        x_size = ((x_size - 2) // 2) + 1
        y_size = ((y_size - 2) // 2) + 1

        # conv2 -> nn.Conv2d
        x_size = (x_size - self.kernel1) + 1
        y_size = (y_size -self.kernel2) + 1
        # conv2 -> nn.MaxPool2d(2)
        x_size = ((x_size - 2) // 2) + 1
        y_size = ((y_size - 2) // 2) + 1

        return x_size * y_size * self.filter2


class CNN_Decoder(nn.Module):
    def __init__(self, max_len, encoding_dim, model_params=None):
        super().__init__()
        self.max_len = max_len
        self.encoding_dim = encoding_dim
        self.dropout_dec = model_params['dropout_dec'] if model_params else 0

        self.fc1 = nn.Sequential(
            nn.Linear(self.encoding_dim, 512),
            nn.ELU(),
            nn.Dropout(self.dropout_dec),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Dropout(self.dropout_dec),
            nn.Linear(1024, self.max_len * 20)   # self.max_len * 21
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CNN_AE(nn.Module):
    def __init__(self, max_len, encoding_dim, batch_size, model_params=None):
        super().__init__()

        self.max_len = max_len
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.model_params = model_params

        self.encoder = CNN_Encoder(self.max_len, self.encoding_dim, self.batch_size, self.model_params)
        self.decoder = CNN_Decoder(self.max_len, self.encoding_dim, self.model_params)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class LSTM_Encoder(nn.Module):
    def __init__(self, num_features, embedding_size, model_params=None):
        super().__init__()

        self.num_features = num_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.LSTM1 = nn.LSTM(num_features,
                             embedding_size,
                             batch_first=True,
                             num_layers=model_params['num_layers_enc'] if model_params else 2,
                             dropout=model_params['dropout_enc'] if model_params else 0.09)

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x = x.view(x.size(0), -1, self.num_features)
        seq_len = x.size(1)
        self.LSTM1.flatten_parameters()
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x[:, -1, :]
        return x, seq_len


class LSTM_Decoder(nn.Module):
    def __init__(self, num_features, output_size, model_params=None):
        super().__init__()

        # self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_size = (2 * num_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=model_params['num_layers_dec'] if model_params else 2,
            dropout=model_params['dropout_dec'] if model_params else 0.02
        )

        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1)
        self.LSTM1.flatten_parameters()
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = self.fc(x)
        x = F.softmax(x, dim=2)
        x = x.view(x.size(0), -1)
        return x


class LSTM_AE(nn.Module):
    def __init__(self, num_features, encoding_dim, model_params=None):
        super().__init__()

        self.num_features = num_features
        self.encoding_dim = encoding_dim
        self.model_params = model_params

        self.encoder = LSTM_Encoder(self.num_features, self.encoding_dim, self.model_params)
        self.decoder = LSTM_Decoder(self.encoding_dim, self.num_features, self.model_params)

    def forward(self, x):
        # torch.manual_seed(0)
        encoded, seq_len = self.encoder(x)
        decoded = self.decoder(encoded, seq_len)
        return encoded, decoded


class HLA_Pep_Model(nn.Module):
    def __init__(self, encoding_dim_pep, encoding_dim_hla, pep_AE, hla_AE,
                 concat_type, concat_oneHot_dim, concat_emb_dim, model_params):
        super().__init__()

        self.hidden_size = model_params['hidden_size']
        self.activ_func = model_params['activ_func']
        self.dropout = model_params['dropout']
        self.encoding_dim_pep = encoding_dim_pep
        self.encoding_dim_hla = encoding_dim_hla
        self.hla_AE = hla_AE
        self.pep_AE = pep_AE
        self.concat_type = concat_type
        self.concat_oneHot_dim = concat_oneHot_dim if 'oneHot' in self.concat_type else 0
        self.concat_emb_dim = concat_emb_dim if self.concat_type == 'emb' else 0
        if self.concat_type == 'emb':
            self.concat_embedding = nn.Embedding(concat_oneHot_dim + 1, concat_emb_dim)

        self.encoded_layer_size = self.encoding_dim_pep + self.encoding_dim_hla + \
                                  self.concat_oneHot_dim + self.concat_emb_dim  # at least one from the 2 lasts is 0

        self.fc1 = nn.Sequential(
            nn.Linear(self.encoded_layer_size, self.hidden_size),
            self.activ_func,
            nn.Dropout(self.dropout),
        )
        self.fc2A = nn.Linear(self.hidden_size, 1)
        self.fc2B = nn.Linear(self.hidden_size, 1)

    def forward(self, pep, hla, hla_oneHot, emb):
        pep_encoded, _ = self.pep_AE(pep)
        hla_encoded, _ = self.hla_AE(hla)

        if self.concat_type == 'emb':
            emb = self.concat_embedding(emb.long())

        if self.concat_type == 'oneHot':
            x = torch.cat((pep_encoded, hla_encoded, hla_oneHot), dim=1)
        elif self.concat_type == 'emb':
            x = torch.cat((pep_encoded, hla_encoded, emb), dim=1)
        elif self.concat_type == 'None':
            x = torch.cat((pep_encoded, hla_encoded), dim=1)
        elif self.concat_type == 'oneHot&zero':
            x = torch.cat((pep_encoded, self.zero_hla_seq(hla_encoded, hla_oneHot), hla_oneHot), dim=1)

        x = self.fc1(x)
        y1 = self.fc2A(x)
        y2 = self.fc2B(x)

        return y1, y2

    @staticmethod
    def zero_hla_seq(hla_encoded, hla_oneHot):
        for i in range(hla_oneHot.size()[0]):
            if torch.count_nonzero(hla_oneHot[i]).item() != 0:  # hla with high freq
                hla_encoded[i] = torch.zeros(hla_encoded[i].size())
        return hla_encoded
