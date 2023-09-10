import sys
import pyopenjtalk
import re
#from dict_pheme_kana import phome2kana_dict, kana2phome_dict

import pyqtgraph as pg
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import random

import pyaudio
import numpy as np
from scipy.io import wavfile
import torch
from torchaudio.compliance import kaldi

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wave
# 音声区間検出及びノイズ除去のためのモジュール
import librosa
import scipy.signal

from xvector_jtubespeech import XVector
from tts_implementation.contrib import Tacotron2PWGTTS
import os

# UMAPを表示するためのmatplotlib
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg

s = np.load("./umap_embedding.npy")
x_data = s[:, 0]
y_data = s[:, 1]

x_data_max = np.max(x_data)
print(x_data_max)
x_data_min = np.min(x_data)
print(x_data_min)
y_data_max = np.max(y_data)
print(y_data_max)
y_data_min = np.min(y_data)
print(y_data_min)

StyleSheet = '''
QPushButton {
    width: 150;
    height: 150;
    border-radius : 75;
    color: white;
    background-color: rgb(100, 0, 0)
}
QPushButton:pressed {
    background-color: rgb(0, 0, 100)
}
'''

StyleSheet_ListWidget = '''
QListWidget{
    border : 2px solid black;
    background : lightgreen;
}
QListWidget QScrollBar{
    background : lightblue;
}
QListView::item:selected{
    border : 2px solid black;
    background : green;
}
'''

"""
音素
"""

# 音素 (+pau/sil)
phonemes = [
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
    "pau",
    "sil",
]

extra_symbols = [
    "^",  # 文の先頭を表す特殊記号 <SOS>
    "$",  # 文の末尾を表す特殊記号 <EOS> (通常)
    "?",  # 文の末尾を表す特殊記号 <EOS> (疑問系)
    "_",  # ポーズ
    "#",  # アクセント句境界
    "[",  # ピッチの上がり位置
    "]",  # ピッチの下がり位置
]

_pad = "~"

# NOTE: 0 をパディングを表す数値とする
symbols = [_pad] + extra_symbols + phonemes


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}



# 正規表現によって数値特徴を取り出す
def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))

def pp_symbols(labels, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels
    The algorithm is based on [Kurihara 2021] [1]_ with some tweaks.
    Args:
        labels (HTSLabelFile): List of labels
        drop_unvoiced_vowels (bool): Drop unvoiced vowels. Defaults to True.
    Returns:
        list: List of phoneme + prosody symbols
    .. ipython::
        In [11]: import ttslearn
        In [12]: from nnmnkwii.io import hts
        In [13]: from ttslearn.tacotron.frontend.openjtalk import pp_symbols
        In [14]: labels = hts.load(ttslearn.util.example_label_file())
        In [15]: " ".join(pp_symbols(labels.contexts))
        Out[15]: '^ m i [ z u o # m a [ r e ] e sh i a k a r a ... $'
    .. [1] K. Kurihara, N. Seiyama, and T. Kumano, “Prosodic features control by
        symbols as input of sequence-to-sequence acoustic modeling for neural tts,”
        IEICE Transactions on Information and Systems, vol. E104.D, no. 2,
        pp. 302–311, 2021.
    """
    PP = []
    N = len(labels)

    # 各音素毎に順番に処理
    for n in range(N):
        lab_curr = labels[n]

        # 当該音素
        # - ←こいつとこいつ→+の間にある文字列パターンを抜き出す＝つまり当該音素
        # 例.こんにちわ 0 番目 :  xx^xx-sil+k=o/A:xx+xx+xx → sil
        #             1 番目 :  xx^sil-k+o=N/A:-4+1+5/B:xx-xx_xx/ → k
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1) # type: ignore

        # 無声化母音を通常の母音として扱う
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()
        
        # 先頭と末尾の sil のみ例外対応
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                PP.append("^")
            elif n == N - 1:
                # 疑問系かどうか
                #通常 E:5_5!0_xx-xx 疑問系 E:5_5!1_xx-xx
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("$")
                elif e3 == 1:
                    PP.append("?")
            continue
        elif p3 == "pau":
            PP.append("_")
            continue
        else:
            PP.append(p3)
        

        # アクセント型及び位置情報(前方または後方)
        # A: 0 から(-) 9　または "-" の一回以上の繰り返し +
        # 例. 0 番目 :  xx^xx-sil+k=o/A:xx+xx+xx/B:  1 番目 :  xx^sil-k+o=N/A:-4+1+5/B
        # 0 番目 :  -50 = None                       1 番目 :  -4
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr) # アクセント核と当該モーラの位置の差
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr) # 当該アクセント句中の当該モーラの位置(前方向)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr) # 当該アクセント句中の当該モーラの位置(後方向)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

        # アクセント句境界
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            PP.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            PP.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            PP.append("[")
    
    return PP




"""
#in_feats = text_to_sequence(pp_symbols(labels))
text = "ジイジが、ヂになる"

# extract_fullcontextによってフルコンテキストラベルを取り出す
labels = pyopenjtalk.extract_fullcontext(text)
phoneme = pp_symbols(labels)
print(phoneme)
"""

def extract_xvector(
  model, # xvector model
  wav   # 16kHz mono
):
  # extract mfcc
  wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
  mfcc = kaldi.mfcc(wav, num_ceps=24, num_mel_bins=24) # [1, T, 24]
  mfcc = mfcc.unsqueeze(0)

  # extract xvector
  xvector = model.vectorize(mfcc) # (1, 512)
  xvector = xvector.to("cpu").detach().numpy().copy()[0]

  return xvector

# pyaudioの速度を直すための関数
def delay(inp, rate):
    outp = []

    for i in range(len(inp)):
        for j in range(rate):
            outp.append(inp[i])

    return np.array(outp)


p = pyaudio.PyAudio()
# 音声の定数
RATE = 16000
CHUNK = 1024
CHANNEL_IN = 1
CHANNEL_OUT = 2
dtype=np.int16
p_output_channels = 1

stream_out = p.open(
                format=pyaudio.paInt16,
                channels = CHANNEL_OUT,
                rate=RATE,
                output_device_index = p_output_channels,
                frames_per_buffer=CHUNK,
                input=False,
                output=True,
)

class Umap2Xvec(nn.Module):
    def __init__(self):
        super(Umap2Xvec, self).__init__()
        self.l1 = nn.Linear(2, 16)
        self.l2 = nn.Linear(16, 64)
        self.l3 = nn.Linear(64, 256)
        self.l4 = nn.Linear(256, 512)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.features = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout,
            self.l2,
            self.relu,
            self.dropout,
            self.l3,
            self.relu,
            self.dropout,
            self.l4
        )
    
    def forward(self, x):
        x1 = self.features(x)
        return x1

Decoder = Umap2Xvec().to("cpu")
Decoder.load_state_dict(torch.load('1000_model.pth', map_location=torch.device('cpu')))
#Decoder.load_state_dict(torch.load('800_model.pth', map_location=torch.device('cpu')))

def cos_sim(xvector, previous_xvector):
    now_xvec_numpy = xvector.to('cpu').detach().numpy().copy()
    pre_xvec_numpy = previous_xvector.to('cpu').detach().numpy().copy()

    now_xvec_numpy_nr = np.linalg.norm(now_xvec_numpy, ord=2)
    pre_xvec_numpy_nr = np.linalg.norm(pre_xvec_numpy, ord=2)

    sim = np.dot(now_xvec_numpy, pre_xvec_numpy) / (now_xvec_numpy_nr * pre_xvec_numpy_nr)
    print("コサイン類似度", sim)

    return sim
        

new_point = None
chara_dict = {}

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        width_gui = 1920
        height_gui = 1200
        self.setGeometry(0, 0, width_gui, height_gui)
        self.setWindowTitle("音声合成アプリ")
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.p = self.palette()
        self.p.setColor(self.backgroundRole(), QColor("#000"))
        self.p.setColor(self.foregroundRole(), QColor("#FFF"))
        self.setPalette(self.p)

        self.textbox = QLineEdit(self)
        self.textbox.move(0, 30)
        self.textbox.resize(width_gui/3, height_gui/50)

        self.textDicideButton = QPushButton("合成", self)
        self.textDicideButton.move(15, 30 + height_gui/30 + 10)
        self.textDicideButton.resize(50, 50)
        self.textDicideButton.clicked.connect(self.make_accent)

        self.wavInputButton = QPushButton("Wav", self)
        self.wavInputButton.move(15 + 50, 30 + height_gui/30 + 10)
        self.wavInputButton.resize(50, 50)
        self.wavInputButton.clicked.connect(self.wavInputFunc)

        self.wavLabel = QLabel("None", self)
        self.wavLabel.move(15 + 100, 30 + height_gui/30 + 10)
        self.wavLabel.resize(140, 50)
        self.xvector = ""

        
        self.graph = pg.PlotWidget(self)
        self.graph.move(0, height_gui/2)
        #self.graph.resize(width_gui/3, height_gui/6)
        self.graph.setMinimumSize(width_gui/3, 100)
        self.graph.setMaximumSize(width_gui/3, 100)
        self.graph_Items = self.graph.plotItem
        self.graph_Items.setYRange(0, 1)


        self.ax_bottom = self.graph.getAxis("bottom")
        self.ax_left = self.graph.getAxis("left")
        self.ax_left.setPen(pg.mkPen(color="#000000"))
        y_labels = [(0, " "), (1, " ")]
        self.ax_bottom.setTicks([(0, " "), (1, " ")])
        self.ax_left.setTicks([y_labels])

        # アクセント修正

        self.accent_repaire_edit = QTextEdit(self)
        self.accent_repaire_edit.move(50, height_gui/1.8 + 100)
        self.accent_repaire_edit.resize(width_gui/4, 30)
        self.accent_repaire_edit.textChanged.connect(self.accentRepairePosition)


        self.accent_repaire_synthe = QPushButton("合成", self)
        self.accent_repaire_synthe.move(0, height_gui/1.8 + 100)
        self.accent_repaire_synthe.resize(50, 50)
        self.accent_repaire_synthe.clicked.connect(self.repaireAccentSynthesis)

        self.accent_flag = False



        """
        self.scroll_accent = QScrollBar(self)
        self.scroll_accent.move(0, height_gui-20)
        self.scroll_accent.setStyleSheet("background-color: #fffff0")
        self.scroll_accent.setOrientation(Qt.Orientation.Horizontal) #横にする
        self.scroll_accent.setRange(0, 100)
        self.scroll_accent.resize(width_gui, 20)
        self.scroll_accent.setValue(0)
        self.scroll_accent.valueChanged.connect(self.scroll_gui)
        """

        """
        self.scroll_button = QPushButton("ゆかり", self)
        self.scroll_button.resize(30, 30)
        self.scroll_button.setStyleSheet(StyleSheet)
        self.scroll_button2 = QPushButton("あかり", self)
        self.scroll_button2.resize(30, 30)
        self.scroll_button2.setStyleSheet(StyleSheet)

        self.scroll = QScrollArea(self)
        self.scroll.move(0, height_gui/2)
        self.scroll.resize(width_gui/3, height_gui/6)
        self.scroll.setWidget(self.scroll_button)
        self.scroll.setWidget(self.scroll_button2)
        """

        # FigureCanvasに表示するグラフ
        fig = Figure()
        #fig.xlim(x_data_min, x_data_max)
        #fig.ylim(y_data_min, y_data_max)
        # グラフを表示するFigureCanvasを作成
        self.fc = FigureCanvas(fig)

        # グラフの設定
        self.fc.axes = fig.add_subplot(1,1,1)
        #self.fc.axes = fig.add_subplot(x_data_min, x_data_max, y_data_min, y_data_max)
        ## [0, 118, 236, 354, 472]
        
        #self.fc.axes.plot([x_data_min, x_data_max, x_data_max, x_data_min, x_data_min], [y_data_min, y_data_min, y_data_max, y_data_max, y_data_min], 'r')
        """
        for i in range(0, len(x_data)):
            if i == 0 or i == 118 or i == 236 or i == 354 or i == 472:
                continue
            self.fc.axes.scatter(x_data[i], y_data[i], color="purple")
        """
        #self.fc.axes.scatter(x_data[0], y_data[0], color="red")
        #self.fc.axes.scatter(x_data[118], y_data[118], color="blue")
        #self.fc.axes.scatter(x_data[236], y_data[236], color="green")
        #self.fc.axes.scatter(x_data[354], y_data[354], color="black")
        #self.fc.axes.scatter(x_data[472], y_data[472], color="pink")


        #####
        #####
        #類似度計算で島つくる
        #####
        #####

        #x_list = [-75,-50,-25,0,25,50,75]
        x_list = [-60,-40,-20,0,20,40,60]
        #y_list = [-75,-50,-25,0,25,50,75]
        y_list = [-60,-40,-20,0,20,40,60]

        device = "cpu"

        for idy in range(0, len(y_list)-1):
            for idx in range(0, len(x_list)-1):
                x0y0 = np.stack([x_list[idx], y_list[idy]])
                source_data = torch.from_numpy(x0y0).type('torch.FloatTensor').to(device)
                x0y0vector = Decoder(source_data)

                x1y0 = np.stack([x_list[idx+1], y_list[idy]])
                source_data = torch.from_numpy(x1y0).type('torch.FloatTensor').to(device)
                x1y0vector = Decoder(source_data)

                x1y1 = np.stack([x_list[idx+1], y_list[idy+1]])
                source_data = torch.from_numpy(x1y1).type('torch.FloatTensor').to(device)
                x1y1vector = Decoder(source_data)

                x0y1 = np.stack([x_list[idx], y_list[idy+1]])
                source_data = torch.from_numpy(x0y1).type('torch.FloatTensor').to(device)
                x0y1vector = Decoder(source_data)

                sikiiti = 0.985
                sim = cos_sim(x0y0vector, x1y0vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx], x_list[idx+1]], [y_list[idy], y_list[idy]], 'black')
                
                sim = cos_sim(x1y0vector, x1y1vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx+1], x_list[idx+1]], [y_list[idy], y_list[idy+1]], 'black')

                sim = cos_sim(x1y1vector, x0y1vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx+1], x_list[idx]], [y_list[idy+1], y_list[idy+1]], 'black')

                sim = cos_sim(x0y1vector, x0y0vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx], x_list[idx]], [y_list[idy+1], y_list[idy]], 'black')

                sim = cos_sim(x0y0vector, x1y1vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx], x_list[idx+1]], [y_list[idy], y_list[idy+1]], 'black')
                
                sim = cos_sim(x1y0vector, x0y1vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx+1], x_list[idx]], [y_list[idy], y_list[idy+1]], 'black')


        # グラフのMAX-MIN
        self.fc.axes.scatter(65, 65, color="white")
        self.fc.axes.scatter(-65, -65, color="white")

        # 描画設定
        self.fc.setParent(self)
        self.fc.move(0, 150)
        self.fc.resize(width_gui / 3, height_gui / 3)

        self.fc.mpl_connect('button_press_event', self.touch_graph)



        self.engine = Tacotron2PWGTTS(model_dir="./tts_models/jvs001-100_sr16000_SV2TTS_parallel_wavegan_sr16k")


        ######
        ######  シナリオ作成UI (右側)
        ######

        self.senarioListBox = QListWidget(self)
        self.senarioListBox.move(width_gui/3 + 70, 30)
        self.senarioListBox.resize(width_gui/3, height_gui/1.8)
        self.senarioListBox.setStyleSheet(StyleSheet_ListWidget)
        self.senarioListBox.setFont(QFont("ＭＳ 明朝", 18))

        #self.senarioListBox.itemClicked.connect(self.senarioVoiceConf)
        self.senarioListBox.itemClicked.connect(self.senarioVoiceConf)
        self.senarioListBox.itemDoubleClicked.connect(self.editOrClear)


        self.senario_character_register = QPushButton('人物登録', self)
        self.senario_character_register.move(width_gui/3 + 70, height_gui/1.8 + 70)
        self.senario_character_register.resize(100, 30)
        self.senario_character_register.clicked.connect(self.CharacterRegister)

        self.senario_character_voice_register = QPushButton('声登録', self)
        self.senario_character_voice_register.move(width_gui/3 + 170, height_gui/1.8 + 70)
        self.senario_character_voice_register.resize(100, 30)
        self.senario_character_voice_register.clicked.connect(self.CharacterVoiceRegister)

        self.senario_character = QComboBox(self)
        self.senario_character.move(width_gui/3 + 70, height_gui/1.8 + 100)
        self.senario_character.resize(100, 30)

        self.senario_character_all = []

        self.senario_serif = QTextEdit(self)
        self.senario_serif.move(width_gui/3+170, height_gui/1.8 + 100)
        self.senario_serif.resize(width_gui/4, 30)

        self.senarioButton = QPushButton("登録", self)
        self.senarioButton.move(width_gui/3+width_gui/4+170, height_gui/1.8 + 100)
        self.senarioButton.resize(50, 30)
        self.senarioButton.clicked.connect(self.registerSenario)


        self.character_list = []
        self.serif_list = []


        #self.textbox.textChanged.connect(self.make_accent)

    def make_accent(self):
        self.graph_Items.clear()
        try:
            phoneme = pp_symbols(pyopenjtalk.extract_fullcontext(self.textbox.text()))
            #print("フルコン",pyopenjtalk.extract_fullcontext(self.textbox.text()))
            print("音素",phoneme)

            self.accent_flag = True
            self.accent_repaire_edit.setText("".join(phoneme[1:-1]))
            #self.scroll_accent.setRange(0, len(katakana) - 1)
            #self.scroll_accent.setValue(0)
            kata = "".join(phoneme)
            x_list = []
            y_list = []
            x_labels = []
            accent_position = 0
            for i in range(1, len(phoneme) - 1):
                if phoneme[i] == "#":
                    if 1 not in y_list:
                        y_list[0] = 1
                    self.graph_Items.addItem(pg.PlotCurveItem(
                    x_list,
                    y_list
                    ))
                    x_list = []
                    y_list = []
                    continue
                if phoneme[i] == "[": 
                    accent_position = 1
                    continue
                if phoneme[i] == "]":
                    accent_position = 0
                    continue

                x_list.append(i)
                y_list.append(accent_position)
                x_labels.append((i, phoneme[i]))
            if 1 not in y_list:
                 y_list[0] = 1

            self.graph_Items.addItem(pg.PlotCurveItem(
                x_list,
                y_list
            ))
            
            self.ax_bottom.setTicks([x_labels])
            self.accent_flag = False
        except Exception as e:
            print(e)
            print("エラーだよい")



        # 音声合成
        if self.xvector == "":
            self.wavLabel.setText("Wavファイルを入れて")
            return
        text = str(self.textbox.text())
        wav, sr = self.engine.tts(text, tqdm=None, spk_id=self.xvector)
        wav = delay(wav, 2)
        stream_out.write(wav.astype(dtype).tobytes())
        
        """
        text = ['^', 'a', 'r', 'a', '[', 'y', 'u', ']', 'r', 'u', 'g', 'e', 'N', '[', 'j', 'i', 'ts', 'u', 'o', '#', 's', 'u', ']', 'b', 'e', 't', 'e', '#', 'j', 'i', '[', 'b', 'u', 'N', 'n', 'o', '#', 'h', 'o', ']', 'o', 'e', '#', 'n', 'e', '[', 'j', 'i', 'm', 'a', 'g', 'e', ']', 't', 'a', '#', 'n', 'o', '[', 'd', 'a', '$']
        
        wav, sr = self.engine.tts(text, tqdm=None, spk_id=self.xvector, phoneme=True)
        wav = delay(wav, 2)
        stream_out.write(wav.astype(dtype).tobytes())
        """
        
        #self.label.setText("".join(phoneme)+"\n"+kata)
        
    """
    def scroll_gui(self):
        print(self.scroll_accent.value())
    """
    
    def touch_graph(self, event):
        global new_point
        if event.button == 1:
            # MouseButton.LEFT
            print('Left Button')
            print('x = ', str(event.xdata))
            print(type(event.xdata))
            print('y = ', str(event.ydata))

            x_y = np.stack([event.xdata, event.ydata])
            #print(x_y)
            #print(x_y.shape)
            source_data = torch.from_numpy(x_y).type('torch.FloatTensor')
            self.xvector = Decoder(source_data)
            self.wavLabel.setText("x:"+str(event.xdata)+"\ny:"+str(event.ydata))
            wav, sr = self.engine.tts("合成", tqdm=None, spk_id=self.xvector)
            wav = delay(wav, 2)
            stream_out.write(wav.astype(dtype).tobytes())

            if new_point is not None:
                new_point.remove()

            new_point = self.fc.axes.scatter(event.xdata, event.ydata, color="blue")
            self.fc.draw()



        elif event.button == 3:
            # MouseButton.RIGHT:
            print('Right Button')
        print("タッチグラフ")
    
    def wavInputFunc(self):
        self.wavLabel.setText("処理中")
        if os.name == "nt":
            filepath = QFileDialog.getOpenFileName(self, 'Open file', "C:\\", "Audio files (*.wav)")
        elif os.name == "posix":
            filepath = QFileDialog.getOpenFileName(self, 'Open file', "/home", "Audio files (*.wav)")
        #print(filepath[0])
        #print(type(filepath[0]))
        _sr, wav_file = wavfile.read(filepath[0]) # 16kHz mono

        if wav_file.dtype in [np.int16, np.int32]:
            wav_file = (wav_file / np.iinfo(wav_file.dtype).max).astype(np.float64)
        wav_file = librosa.resample(wav_file, _sr, 16000)
        #wav_file, _ = librosa.effects.trim(wav_file, top_db=25)
        #wav_file = scipy.signal.wiener(wav_file)
        model = XVector("xvector.pth")
        self.xvector = extract_xvector(model, wav_file) # (512, )
        self.wavLabel.setText(os.path.basename(filepath[0]))
    
    def CharacterRegister(self):
        # ダイアログ表示
        text, ok = QInputDialog.getText(self, '--INPUT CHARACTER NAME', 'Enter Character Name')

        if ok:
            self.senario_character.addItem(text)
            self.senario_character_all.append(text)
    
    def CharacterVoiceRegister(self):
        if self.xvector == "":
            self.wavLabel.setText("話者を選択してや")
            return
        chara_dict[self.senario_character.currentText()] = self.xvector

    
    def registerSenario(self):
        print("--")
        """
        ls = [self.senario_character.currentText(),
        self.senario_serif.toPlainText()
        ]
        self.senarioListBox.addItems(ls)
        """
        self.serif_list.append(self.senario_serif.toPlainText())
        self.character_list.append(self.senario_character.currentText())
        self.senarioListBox.addItem(self.senario_character.currentText() + " : " + self.senario_serif.toPlainText())
        self.senario_serif.clear()

        #self.senarioListBox.setStyleSheet(StyleSheet_ListWidget)
    
    def editOrClear(self, item):
        qm = QMessageBox(self) #削除の確認のためメッセージボックスを表示
        qm.setText("編集or削除")
        cancelbutton = qm.addButton("キャンセル", QMessageBox.ActionRole)
        editbutton = qm.addButton("編集", QMessageBox.ActionRole)
        clearbutton = qm.addButton("削除", QMessageBox.ActionRole)
        qm.setDefaultButton(editbutton)
        qm.exec_()

        if qm.clickedButton() == editbutton:
            print("編集")
            self.editScenario()
        elif qm.clickedButton() == clearbutton:
            print("削除")
            self.clearScenario()
        elif qm.clickedButton() == cancelbutton:
            print("キャンセル")
        
        
    
    def editScenario(self):
        print("edit")

        chara, ok = QInputDialog.getItem(self, "キャラ編集", 
        "前:"+str(self.character_list[self.senarioListBox.currentRow()]),
        self.senario_character_all
        )
        text, ok2 = QInputDialog.getText(self, '編集', '前 : ' + str(self.serif_list[self.senarioListBox.currentRow()]))

        if ok:
            self.character_list[self.senarioListBox.currentRow()] = chara
        
        if ok2:
            self.serif_list[self.senarioListBox.currentRow()] = text
        self.senarioListBox.clear()
        for i in range(0, len(self.character_list)):
            self.senarioListBox.addItem(self.character_list[i] + " : " + self.serif_list[i])


    
    def clearScenario(self):
        #print(self.senarioListBox.currentRow())
        self.character_list.pop(self.senarioListBox.currentRow())
        self.serif_list.pop(self.senarioListBox.currentRow())
        self.senarioListBox.clear()
        for i in range(0, len(self.character_list)):
            self.senarioListBox.addItem(self.character_list[i] + " : " + self.serif_list[i])

    
    def senarioVoiceConf(self, item):
        print("シナリオクリック")
        text = str(item.text())
        print(text)
        index = 0
        for i in range(0, len(text)):
            if text[i] == ":":
                index = i
                break
        
        gosei_washa = text[0:i-1]
        gosei_text = text[i+2:len(text)]
        print(gosei_text)
        print(gosei_washa)
        wav, sr = self.engine.tts(str(gosei_text), tqdm=None, spk_id=chara_dict[str(gosei_washa)])
        wav = delay(wav, 2)
        stream_out.write(wav.astype(dtype).tobytes())
    

    def repaireAccentSynthesis(self):
        print("合成")
        # 音声合成
        if self.xvector == "":
            self.wavLabel.setText("Wavファイルを入れて")
            return
        
        phoneme = pp_symbols(pyopenjtalk.extract_fullcontext(self.textbox.text()))
        text = []
        text.append('^')
        #accent = str(self.accent_repaire_edit.toPlainText())
        accent = str(self.accent_repaire_edit.toPlainText())

        skip = []

        # chみたいな音素が c hと別れてしまう
        # []#の場所を探索して移し替える
        for i in range(0, len(accent)):
            if i in skip:
                continue
            #"by","ch","cl","dy","gy","hy","ky","my","ny","py","ry","sh","ts","ty",
            if  accent[i] == "[" or accent[i] == "["  or accent[i] == "#":
                if i != 0:
                    if not (accent[i-1] == "a" or accent[i-1] == "i" or accent[i-1] == "u" or accent[i-1] == "e" or accent[i-1] == "o" or accent[i-1] == "N"):
                        continue
                
            if accent[i] == "b" or accent[i] == "c"  or accent[i] == "d" or accent[i] == "g" or accent[i] == "h" or accent[i] == "k" or accent[i] == "m" or accent[i] == "n" or accent[i] == "p" or accent[i] == "r" or accent[i] == "s" or accent[i] == "t":
                if i != len(accent):
                    if accent[i+1] == "[" or accent[i+1] == "["  or accent[i+1] == "#":
                        if i+1 == len(accent):
                            continue
                        if accent[i+2] == "y" or accent[i+2] == "h" or accent[i+2] == "l" or accent[i+2] == "s":
                            #i += 2
                            for j in range(i+1, i+3):
                                skip.append(j)
                            text.append(accent[i]+accent[i+2])
                            continue
                    if accent[i+1] == "y" or accent[i+1] == "h" or accent[i+1] == "l" or accent[i+1] == "s":
                        #i += 1
                        for j in range(i+1, i+2):
                            skip.append(j)
                        text.append(accent[i]+accent[i+1])
                        continue
            text.append(accent[i])
        text.append('$')

        #for i in range(0, len(accent)):
            #text.append(accent[i])
        #text.append('$')
        print(text)
        print(type(text))
        wav, sr = self.engine.tts(text, tqdm=None, spk_id=self.xvector, phoneme=True)
        wav = delay(wav, 2)
        stream_out.write(wav.astype(dtype).tobytes())
    
    def accentRepairePosition(self):
        print("アクセント位置")
        if self.accent_flag == True:
            return

        try:
            self.graph_Items.clear()
            #phoneme = pp_symbols(pyopenjtalk.extract_fullcontext(self.textbox.text()))
            phoneme = ['^']
            accent = self.accent_repaire_edit.toPlainText()

            print(accent)
            skip = []

            for i in range(0, len(accent)):
                if i in skip:
                    continue
                #"by","ch","cl","dy","gy","hy","ky","my","ny","py","ry","sh","ts","ty",
                if  accent[i] == "[" or accent[i] == "["  or accent[i] == "#":
                    if i != 0:
                        if not (accent[i-1] == "a" or accent[i-1] == "i" or accent[i-1] == "u" or accent[i-1] == "e" or accent[i-1] == "o" or accent[i-1] == "N"):
                            continue
                        
                
                if accent[i] == "b" or accent[i] == "c"  or accent[i] == "d" or accent[i] == "g" or accent[i] == "h" or accent[i] == "k" or accent[i] == "m" or accent[i] == "n" or accent[i] == "p" or accent[i] == "r" or accent[i] == "s" or accent[i] == "t":
                    if i != len(accent):
                        if accent[i+1] == "[" or accent[i+1] == "["  or accent[i+1] == "#":
                            if i+1 == len(accent):
                                continue
                            if accent[i+2] == "y" or accent[i+2] == "h" or accent[i+2] == "l" or accent[i+2] == "s":
                                #i += 2
                                for j in range(i+1, i+3):
                                    skip.append(j)
                                print(accent[i]+accent[i+2])
                                print(i)
                                phoneme.append(accent[i]+accent[i+2])
                                continue
                        if accent[i+1] == "y" or accent[i+1] == "h" or accent[i+1] == "l" or accent[i+1] == "s":
                            #i += 1
                            for j in range(i+1, i+2):
                                skip.append(j)
                            print(accent[i]+accent[i+1])
                            print(i)
                            phoneme.append(accent[i]+accent[i+1])
                            continue
                phoneme.append(accent[i])
                print(phoneme)
                print(i)
            phoneme.append('$')
            #print("フルコン",pyopenjtalk.extract_fullcontext(self.textbox.text()))
            print("音素",phoneme)

            #self.scroll_accent.setRange(0, len(katakana) - 1)
            #self.scroll_accent.setValue(0)
            kata = "".join(phoneme)
            x_list = []
            y_list = []
            x_labels = []
            accent_position = 0
            for i in range(1, len(phoneme) - 1):
                if phoneme[i] == "#":
                    if 1 not in y_list:
                        y_list[0] = 1
                    self.graph_Items.addItem(pg.PlotCurveItem(
                    x_list,
                    y_list
                    ))
                    x_list = []
                    y_list = []
                    continue
                if phoneme[i] == "[": 
                    accent_position = 1
                    continue
                if phoneme[i] == "]":
                    accent_position = 0
                    continue

                x_list.append(i)
                y_list.append(accent_position)
                x_labels.append((i, phoneme[i]))
            if 1 not in y_list:
                 y_list[0] = 1

            self.graph_Items.addItem(pg.PlotCurveItem(
                x_list,
                y_list
            ))
            
            self.ax_bottom.setTicks([x_labels])
        except Exception as e:
            print(e)
            print("エラーだよい")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()import sys
import pyopenjtalk
import re
#from dict_pheme_kana import phome2kana_dict, kana2phome_dict

import pyqtgraph as pg
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import random

import pyaudio
import numpy as np
from scipy.io import wavfile
import torch
from torchaudio.compliance import kaldi

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wave
# 音声区間検出及びノイズ除去のためのモジュール
import librosa
import scipy.signal

from xvector_jtubespeech import XVector
from tts_implementation.contrib import Tacotron2PWGTTS
import os

# UMAPを表示するためのmatplotlib
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg

s = np.load("./umap_embedding.npy")
x_data = s[:, 0]
y_data = s[:, 1]

x_data_max = np.max(x_data)
print(x_data_max)
x_data_min = np.min(x_data)
print(x_data_min)
y_data_max = np.max(y_data)
print(y_data_max)
y_data_min = np.min(y_data)
print(y_data_min)

StyleSheet = '''
QPushButton {
    width: 150;
    height: 150;
    border-radius : 75;
    color: white;
    background-color: rgb(100, 0, 0)
}
QPushButton:pressed {
    background-color: rgb(0, 0, 100)
}
'''

StyleSheet_ListWidget = '''
QListWidget{
    border : 2px solid black;
    background : lightgreen;
}
QListWidget QScrollBar{
    background : lightblue;
}
QListView::item:selected{
    border : 2px solid black;
    background : green;
}
'''

"""
音素
"""

# 音素 (+pau/sil)
phonemes = [
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
    "pau",
    "sil",
]

extra_symbols = [
    "^",  # 文の先頭を表す特殊記号 <SOS>
    "$",  # 文の末尾を表す特殊記号 <EOS> (通常)
    "?",  # 文の末尾を表す特殊記号 <EOS> (疑問系)
    "_",  # ポーズ
    "#",  # アクセント句境界
    "[",  # ピッチの上がり位置
    "]",  # ピッチの下がり位置
]

_pad = "~"

# NOTE: 0 をパディングを表す数値とする
symbols = [_pad] + extra_symbols + phonemes


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}



# 正規表現によって数値特徴を取り出す
def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))

def pp_symbols(labels, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels
    The algorithm is based on [Kurihara 2021] [1]_ with some tweaks.
    Args:
        labels (HTSLabelFile): List of labels
        drop_unvoiced_vowels (bool): Drop unvoiced vowels. Defaults to True.
    Returns:
        list: List of phoneme + prosody symbols
    .. ipython::
        In [11]: import ttslearn
        In [12]: from nnmnkwii.io import hts
        In [13]: from ttslearn.tacotron.frontend.openjtalk import pp_symbols
        In [14]: labels = hts.load(ttslearn.util.example_label_file())
        In [15]: " ".join(pp_symbols(labels.contexts))
        Out[15]: '^ m i [ z u o # m a [ r e ] e sh i a k a r a ... $'
    .. [1] K. Kurihara, N. Seiyama, and T. Kumano, “Prosodic features control by
        symbols as input of sequence-to-sequence acoustic modeling for neural tts,”
        IEICE Transactions on Information and Systems, vol. E104.D, no. 2,
        pp. 302–311, 2021.
    """
    PP = []
    N = len(labels)

    # 各音素毎に順番に処理
    for n in range(N):
        lab_curr = labels[n]

        # 当該音素
        # - ←こいつとこいつ→+の間にある文字列パターンを抜き出す＝つまり当該音素
        # 例.こんにちわ 0 番目 :  xx^xx-sil+k=o/A:xx+xx+xx → sil
        #             1 番目 :  xx^sil-k+o=N/A:-4+1+5/B:xx-xx_xx/ → k
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1) # type: ignore

        # 無声化母音を通常の母音として扱う
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()
        
        # 先頭と末尾の sil のみ例外対応
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                PP.append("^")
            elif n == N - 1:
                # 疑問系かどうか
                #通常 E:5_5!0_xx-xx 疑問系 E:5_5!1_xx-xx
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("$")
                elif e3 == 1:
                    PP.append("?")
            continue
        elif p3 == "pau":
            PP.append("_")
            continue
        else:
            PP.append(p3)
        

        # アクセント型及び位置情報(前方または後方)
        # A: 0 から(-) 9　または "-" の一回以上の繰り返し +
        # 例. 0 番目 :  xx^xx-sil+k=o/A:xx+xx+xx/B:  1 番目 :  xx^sil-k+o=N/A:-4+1+5/B
        # 0 番目 :  -50 = None                       1 番目 :  -4
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr) # アクセント核と当該モーラの位置の差
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr) # 当該アクセント句中の当該モーラの位置(前方向)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr) # 当該アクセント句中の当該モーラの位置(後方向)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

        # アクセント句境界
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            PP.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            PP.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            PP.append("[")
    
    return PP




"""
#in_feats = text_to_sequence(pp_symbols(labels))
text = "ジイジが、ヂになる"

# extract_fullcontextによってフルコンテキストラベルを取り出す
labels = pyopenjtalk.extract_fullcontext(text)
phoneme = pp_symbols(labels)
print(phoneme)
"""

def extract_xvector(
  model, # xvector model
  wav   # 16kHz mono
):
  # extract mfcc
  wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
  mfcc = kaldi.mfcc(wav, num_ceps=24, num_mel_bins=24) # [1, T, 24]
  mfcc = mfcc.unsqueeze(0)

  # extract xvector
  xvector = model.vectorize(mfcc) # (1, 512)
  xvector = xvector.to("cpu").detach().numpy().copy()[0]

  return xvector

# pyaudioの速度を直すための関数
def delay(inp, rate):
    outp = []

    for i in range(len(inp)):
        for j in range(rate):
            outp.append(inp[i])

    return np.array(outp)


p = pyaudio.PyAudio()
# 音声の定数
RATE = 16000
CHUNK = 1024
CHANNEL_IN = 1
CHANNEL_OUT = 2
dtype=np.int16
p_output_channels = 1

stream_out = p.open(
                format=pyaudio.paInt16,
                channels = CHANNEL_OUT,
                rate=RATE,
                output_device_index = p_output_channels,
                frames_per_buffer=CHUNK,
                input=False,
                output=True,
)

class Umap2Xvec(nn.Module):
    def __init__(self):
        super(Umap2Xvec, self).__init__()
        self.l1 = nn.Linear(2, 16)
        self.l2 = nn.Linear(16, 64)
        self.l3 = nn.Linear(64, 256)
        self.l4 = nn.Linear(256, 512)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.features = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout,
            self.l2,
            self.relu,
            self.dropout,
            self.l3,
            self.relu,
            self.dropout,
            self.l4
        )
    
    def forward(self, x):
        x1 = self.features(x)
        return x1

Decoder = Umap2Xvec().to("cpu")
Decoder.load_state_dict(torch.load('1000_model.pth', map_location=torch.device('cpu')))
#Decoder.load_state_dict(torch.load('800_model.pth', map_location=torch.device('cpu')))

def cos_sim(xvector, previous_xvector):
    now_xvec_numpy = xvector.to('cpu').detach().numpy().copy()
    pre_xvec_numpy = previous_xvector.to('cpu').detach().numpy().copy()

    now_xvec_numpy_nr = np.linalg.norm(now_xvec_numpy, ord=2)
    pre_xvec_numpy_nr = np.linalg.norm(pre_xvec_numpy, ord=2)

    sim = np.dot(now_xvec_numpy, pre_xvec_numpy) / (now_xvec_numpy_nr * pre_xvec_numpy_nr)
    print("コサイン類似度", sim)

    return sim
        

new_point = None
chara_dict = {}

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        width_gui = 1920
        height_gui = 1200
        self.setGeometry(0, 0, width_gui, height_gui)
        self.setWindowTitle("音声合成アプリ")
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.p = self.palette()
        self.p.setColor(self.backgroundRole(), QColor("#000"))
        self.p.setColor(self.foregroundRole(), QColor("#FFF"))
        self.setPalette(self.p)

        self.textbox = QLineEdit(self)
        self.textbox.move(0, 30)
        self.textbox.resize(width_gui/3, height_gui/50)

        self.textDicideButton = QPushButton("合成", self)
        self.textDicideButton.move(15, 30 + height_gui/30 + 10)
        self.textDicideButton.resize(50, 50)
        self.textDicideButton.clicked.connect(self.make_accent)

        self.wavInputButton = QPushButton("Wav", self)
        self.wavInputButton.move(15 + 50, 30 + height_gui/30 + 10)
        self.wavInputButton.resize(50, 50)
        self.wavInputButton.clicked.connect(self.wavInputFunc)

        self.wavLabel = QLabel("None", self)
        self.wavLabel.move(15 + 100, 30 + height_gui/30 + 10)
        self.wavLabel.resize(140, 50)
        self.xvector = ""

        
        self.graph = pg.PlotWidget(self)
        self.graph.move(0, height_gui/2)
        #self.graph.resize(width_gui/3, height_gui/6)
        self.graph.setMinimumSize(width_gui/3, 100)
        self.graph.setMaximumSize(width_gui/3, 100)
        self.graph_Items = self.graph.plotItem
        self.graph_Items.setYRange(0, 1)


        self.ax_bottom = self.graph.getAxis("bottom")
        self.ax_left = self.graph.getAxis("left")
        self.ax_left.setPen(pg.mkPen(color="#000000"))
        y_labels = [(0, " "), (1, " ")]
        self.ax_bottom.setTicks([(0, " "), (1, " ")])
        self.ax_left.setTicks([y_labels])

        # アクセント修正

        self.accent_repaire_edit = QTextEdit(self)
        self.accent_repaire_edit.move(50, height_gui/1.8 + 100)
        self.accent_repaire_edit.resize(width_gui/4, 30)
        self.accent_repaire_edit.textChanged.connect(self.accentRepairePosition)


        self.accent_repaire_synthe = QPushButton("合成", self)
        self.accent_repaire_synthe.move(0, height_gui/1.8 + 100)
        self.accent_repaire_synthe.resize(50, 50)
        self.accent_repaire_synthe.clicked.connect(self.repaireAccentSynthesis)

        self.accent_flag = False



        """
        self.scroll_accent = QScrollBar(self)
        self.scroll_accent.move(0, height_gui-20)
        self.scroll_accent.setStyleSheet("background-color: #fffff0")
        self.scroll_accent.setOrientation(Qt.Orientation.Horizontal) #横にする
        self.scroll_accent.setRange(0, 100)
        self.scroll_accent.resize(width_gui, 20)
        self.scroll_accent.setValue(0)
        self.scroll_accent.valueChanged.connect(self.scroll_gui)
        """

        """
        self.scroll_button = QPushButton("ゆかり", self)
        self.scroll_button.resize(30, 30)
        self.scroll_button.setStyleSheet(StyleSheet)
        self.scroll_button2 = QPushButton("あかり", self)
        self.scroll_button2.resize(30, 30)
        self.scroll_button2.setStyleSheet(StyleSheet)

        self.scroll = QScrollArea(self)
        self.scroll.move(0, height_gui/2)
        self.scroll.resize(width_gui/3, height_gui/6)
        self.scroll.setWidget(self.scroll_button)
        self.scroll.setWidget(self.scroll_button2)
        """

        # FigureCanvasに表示するグラフ
        fig = Figure()
        #fig.xlim(x_data_min, x_data_max)
        #fig.ylim(y_data_min, y_data_max)
        # グラフを表示するFigureCanvasを作成
        self.fc = FigureCanvas(fig)

        # グラフの設定
        self.fc.axes = fig.add_subplot(1,1,1)
        #self.fc.axes = fig.add_subplot(x_data_min, x_data_max, y_data_min, y_data_max)
        ## [0, 118, 236, 354, 472]
        
        #self.fc.axes.plot([x_data_min, x_data_max, x_data_max, x_data_min, x_data_min], [y_data_min, y_data_min, y_data_max, y_data_max, y_data_min], 'r')
        """
        for i in range(0, len(x_data)):
            if i == 0 or i == 118 or i == 236 or i == 354 or i == 472:
                continue
            self.fc.axes.scatter(x_data[i], y_data[i], color="purple")
        """
        #self.fc.axes.scatter(x_data[0], y_data[0], color="red")
        #self.fc.axes.scatter(x_data[118], y_data[118], color="blue")
        #self.fc.axes.scatter(x_data[236], y_data[236], color="green")
        #self.fc.axes.scatter(x_data[354], y_data[354], color="black")
        #self.fc.axes.scatter(x_data[472], y_data[472], color="pink")


        #####
        #####
        #類似度計算で島つくる
        #####
        #####

        #x_list = [-75,-50,-25,0,25,50,75]
        x_list = [-60,-40,-20,0,20,40,60]
        #y_list = [-75,-50,-25,0,25,50,75]
        y_list = [-60,-40,-20,0,20,40,60]

        device = "cpu"

        for idy in range(0, len(y_list)-1):
            for idx in range(0, len(x_list)-1):
                x0y0 = np.stack([x_list[idx], y_list[idy]])
                source_data = torch.from_numpy(x0y0).type('torch.FloatTensor').to(device)
                x0y0vector = Decoder(source_data)

                x1y0 = np.stack([x_list[idx+1], y_list[idy]])
                source_data = torch.from_numpy(x1y0).type('torch.FloatTensor').to(device)
                x1y0vector = Decoder(source_data)

                x1y1 = np.stack([x_list[idx+1], y_list[idy+1]])
                source_data = torch.from_numpy(x1y1).type('torch.FloatTensor').to(device)
                x1y1vector = Decoder(source_data)

                x0y1 = np.stack([x_list[idx], y_list[idy+1]])
                source_data = torch.from_numpy(x0y1).type('torch.FloatTensor').to(device)
                x0y1vector = Decoder(source_data)

                sikiiti = 0.985
                sim = cos_sim(x0y0vector, x1y0vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx], x_list[idx+1]], [y_list[idy], y_list[idy]], 'black')
                
                sim = cos_sim(x1y0vector, x1y1vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx+1], x_list[idx+1]], [y_list[idy], y_list[idy+1]], 'black')

                sim = cos_sim(x1y1vector, x0y1vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx+1], x_list[idx]], [y_list[idy+1], y_list[idy+1]], 'black')

                sim = cos_sim(x0y1vector, x0y0vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx], x_list[idx]], [y_list[idy+1], y_list[idy]], 'black')

                sim = cos_sim(x0y0vector, x1y1vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx], x_list[idx+1]], [y_list[idy], y_list[idy+1]], 'black')
                
                sim = cos_sim(x1y0vector, x0y1vector)
                if sim > sikiiti:
                    self.fc.axes.plot([x_list[idx+1], x_list[idx]], [y_list[idy], y_list[idy+1]], 'black')


        # グラフのMAX-MIN
        self.fc.axes.scatter(65, 65, color="white")
        self.fc.axes.scatter(-65, -65, color="white")

        # 描画設定
        self.fc.setParent(self)
        self.fc.move(0, 150)
        self.fc.resize(width_gui / 3, height_gui / 3)

        self.fc.mpl_connect('button_press_event', self.touch_graph)



        self.engine = Tacotron2PWGTTS(model_dir="./tts_models/jvs001-100_sr16000_SV2TTS_parallel_wavegan_sr16k")


        ######
        ######  シナリオ作成UI (右側)
        ######

        self.senarioListBox = QListWidget(self)
        self.senarioListBox.move(width_gui/3 + 70, 30)
        self.senarioListBox.resize(width_gui/3, height_gui/1.8)
        self.senarioListBox.setStyleSheet(StyleSheet_ListWidget)
        self.senarioListBox.setFont(QFont("ＭＳ 明朝", 18))

        #self.senarioListBox.itemClicked.connect(self.senarioVoiceConf)
        self.senarioListBox.itemClicked.connect(self.senarioVoiceConf)
        self.senarioListBox.itemDoubleClicked.connect(self.editOrClear)


        self.senario_character_register = QPushButton('人物登録', self)
        self.senario_character_register.move(width_gui/3 + 70, height_gui/1.8 + 70)
        self.senario_character_register.resize(100, 30)
        self.senario_character_register.clicked.connect(self.CharacterRegister)

        self.senario_character_voice_register = QPushButton('声登録', self)
        self.senario_character_voice_register.move(width_gui/3 + 170, height_gui/1.8 + 70)
        self.senario_character_voice_register.resize(100, 30)
        self.senario_character_voice_register.clicked.connect(self.CharacterVoiceRegister)

        self.senario_character = QComboBox(self)
        self.senario_character.move(width_gui/3 + 70, height_gui/1.8 + 100)
        self.senario_character.resize(100, 30)

        self.senario_character_all = []

        self.senario_serif = QTextEdit(self)
        self.senario_serif.move(width_gui/3+170, height_gui/1.8 + 100)
        self.senario_serif.resize(width_gui/4, 30)

        self.senarioButton = QPushButton("登録", self)
        self.senarioButton.move(width_gui/3+width_gui/4+170, height_gui/1.8 + 100)
        self.senarioButton.resize(50, 30)
        self.senarioButton.clicked.connect(self.registerSenario)


        self.character_list = []
        self.serif_list = []


        #self.textbox.textChanged.connect(self.make_accent)

    def make_accent(self):
        self.graph_Items.clear()
        try:
            phoneme = pp_symbols(pyopenjtalk.extract_fullcontext(self.textbox.text()))
            #print("フルコン",pyopenjtalk.extract_fullcontext(self.textbox.text()))
            print("音素",phoneme)

            self.accent_flag = True
            self.accent_repaire_edit.setText("".join(phoneme[1:-1]))
            #self.scroll_accent.setRange(0, len(katakana) - 1)
            #self.scroll_accent.setValue(0)
            kata = "".join(phoneme)
            x_list = []
            y_list = []
            x_labels = []
            accent_position = 0
            for i in range(1, len(phoneme) - 1):
                if phoneme[i] == "#":
                    if 1 not in y_list:
                        y_list[0] = 1
                    self.graph_Items.addItem(pg.PlotCurveItem(
                    x_list,
                    y_list
                    ))
                    x_list = []
                    y_list = []
                    continue
                if phoneme[i] == "[": 
                    accent_position = 1
                    continue
                if phoneme[i] == "]":
                    accent_position = 0
                    continue

                x_list.append(i)
                y_list.append(accent_position)
                x_labels.append((i, phoneme[i]))
            if 1 not in y_list:
                 y_list[0] = 1

            self.graph_Items.addItem(pg.PlotCurveItem(
                x_list,
                y_list
            ))
            
            self.ax_bottom.setTicks([x_labels])
            self.accent_flag = False
        except Exception as e:
            print(e)
            print("エラーだよい")



        # 音声合成
        if self.xvector == "":
            self.wavLabel.setText("Wavファイルを入れて")
            return
        text = str(self.textbox.text())
        wav, sr = self.engine.tts(text, tqdm=None, spk_id=self.xvector)
        wav = delay(wav, 2)
        stream_out.write(wav.astype(dtype).tobytes())
        
        """
        text = ['^', 'a', 'r', 'a', '[', 'y', 'u', ']', 'r', 'u', 'g', 'e', 'N', '[', 'j', 'i', 'ts', 'u', 'o', '#', 's', 'u', ']', 'b', 'e', 't', 'e', '#', 'j', 'i', '[', 'b', 'u', 'N', 'n', 'o', '#', 'h', 'o', ']', 'o', 'e', '#', 'n', 'e', '[', 'j', 'i', 'm', 'a', 'g', 'e', ']', 't', 'a', '#', 'n', 'o', '[', 'd', 'a', '$']
        
        wav, sr = self.engine.tts(text, tqdm=None, spk_id=self.xvector, phoneme=True)
        wav = delay(wav, 2)
        stream_out.write(wav.astype(dtype).tobytes())
        """
        
        #self.label.setText("".join(phoneme)+"\n"+kata)
        
    """
    def scroll_gui(self):
        print(self.scroll_accent.value())
    """
    
    def touch_graph(self, event):
        global new_point
        if event.button == 1:
            # MouseButton.LEFT
            print('Left Button')
            print('x = ', str(event.xdata))
            print(type(event.xdata))
            print('y = ', str(event.ydata))

            x_y = np.stack([event.xdata, event.ydata])
            #print(x_y)
            #print(x_y.shape)
            source_data = torch.from_numpy(x_y).type('torch.FloatTensor')
            self.xvector = Decoder(source_data)
            self.wavLabel.setText("x:"+str(event.xdata)+"\ny:"+str(event.ydata))
            wav, sr = self.engine.tts("合成", tqdm=None, spk_id=self.xvector)
            wav = delay(wav, 2)
            stream_out.write(wav.astype(dtype).tobytes())

            if new_point is not None:
                new_point.remove()

            new_point = self.fc.axes.scatter(event.xdata, event.ydata, color="blue")
            self.fc.draw()



        elif event.button == 3:
            # MouseButton.RIGHT:
            print('Right Button')
        print("タッチグラフ")
    
    def wavInputFunc(self):
        self.wavLabel.setText("処理中")
        if os.name == "nt":
            filepath = QFileDialog.getOpenFileName(self, 'Open file', "C:\\", "Audio files (*.wav)")
        elif os.name == "posix":
            filepath = QFileDialog.getOpenFileName(self, 'Open file', "/home", "Audio files (*.wav)")
        #print(filepath[0])
        #print(type(filepath[0]))
        _sr, wav_file = wavfile.read(filepath[0]) # 16kHz mono

        if wav_file.dtype in [np.int16, np.int32]:
            wav_file = (wav_file / np.iinfo(wav_file.dtype).max).astype(np.float64)
        wav_file = librosa.resample(wav_file, _sr, 16000)
        #wav_file, _ = librosa.effects.trim(wav_file, top_db=25)
        #wav_file = scipy.signal.wiener(wav_file)
        model = XVector("xvector.pth")
        self.xvector = extract_xvector(model, wav_file) # (512, )
        self.wavLabel.setText(os.path.basename(filepath[0]))
    
    def CharacterRegister(self):
        # ダイアログ表示
        text, ok = QInputDialog.getText(self, '--INPUT CHARACTER NAME', 'Enter Character Name')

        if ok:
            self.senario_character.addItem(text)
            self.senario_character_all.append(text)
    
    def CharacterVoiceRegister(self):
        if self.xvector == "":
            self.wavLabel.setText("話者を選択してや")
            return
        chara_dict[self.senario_character.currentText()] = self.xvector

    
    def registerSenario(self):
        print("--")
        """
        ls = [self.senario_character.currentText(),
        self.senario_serif.toPlainText()
        ]
        self.senarioListBox.addItems(ls)
        """
        self.serif_list.append(self.senario_serif.toPlainText())
        self.character_list.append(self.senario_character.currentText())
        self.senarioListBox.addItem(self.senario_character.currentText() + " : " + self.senario_serif.toPlainText())
        self.senario_serif.clear()

        #self.senarioListBox.setStyleSheet(StyleSheet_ListWidget)
    
    def editOrClear(self, item):
        qm = QMessageBox(self) #削除の確認のためメッセージボックスを表示
        qm.setText("編集or削除")
        cancelbutton = qm.addButton("キャンセル", QMessageBox.ActionRole)
        editbutton = qm.addButton("編集", QMessageBox.ActionRole)
        clearbutton = qm.addButton("削除", QMessageBox.ActionRole)
        qm.setDefaultButton(editbutton)
        qm.exec_()

        if qm.clickedButton() == editbutton:
            print("編集")
            self.editScenario()
        elif qm.clickedButton() == clearbutton:
            print("削除")
            self.clearScenario()
        elif qm.clickedButton() == cancelbutton:
            print("キャンセル")
        
        
    
    def editScenario(self):
        print("edit")

        chara, ok = QInputDialog.getItem(self, "キャラ編集", 
        "前:"+str(self.character_list[self.senarioListBox.currentRow()]),
        self.senario_character_all
        )
        text, ok2 = QInputDialog.getText(self, '編集', '前 : ' + str(self.serif_list[self.senarioListBox.currentRow()]))

        if ok:
            self.character_list[self.senarioListBox.currentRow()] = chara
        
        if ok2:
            self.serif_list[self.senarioListBox.currentRow()] = text
        self.senarioListBox.clear()
        for i in range(0, len(self.character_list)):
            self.senarioListBox.addItem(self.character_list[i] + " : " + self.serif_list[i])


    
    def clearScenario(self):
        #print(self.senarioListBox.currentRow())
        self.character_list.pop(self.senarioListBox.currentRow())
        self.serif_list.pop(self.senarioListBox.currentRow())
        self.senarioListBox.clear()
        for i in range(0, len(self.character_list)):
            self.senarioListBox.addItem(self.character_list[i] + " : " + self.serif_list[i])

    
    def senarioVoiceConf(self, item):
        print("シナリオクリック")
        text = str(item.text())
        print(text)
        index = 0
        for i in range(0, len(text)):
            if text[i] == ":":
                index = i
                break
        
        gosei_washa = text[0:i-1]
        gosei_text = text[i+2:len(text)]
        print(gosei_text)
        print(gosei_washa)
        wav, sr = self.engine.tts(str(gosei_text), tqdm=None, spk_id=chara_dict[str(gosei_washa)])
        wav = delay(wav, 2)
        stream_out.write(wav.astype(dtype).tobytes())
    

    def repaireAccentSynthesis(self):
        print("合成")
        # 音声合成
        if self.xvector == "":
            self.wavLabel.setText("Wavファイルを入れて")
            return
        
        phoneme = pp_symbols(pyopenjtalk.extract_fullcontext(self.textbox.text()))
        text = []
        text.append('^')
        #accent = str(self.accent_repaire_edit.toPlainText())
        accent = str(self.accent_repaire_edit.toPlainText())

        skip = []

        # chみたいな音素が c hと別れてしまう
        # []#の場所を探索して移し替える
        for i in range(0, len(accent)):
            if i in skip:
                continue
            #"by","ch","cl","dy","gy","hy","ky","my","ny","py","ry","sh","ts","ty",
            if  accent[i] == "[" or accent[i] == "["  or accent[i] == "#":
                if i != 0:
                    if not (accent[i-1] == "a" or accent[i-1] == "i" or accent[i-1] == "u" or accent[i-1] == "e" or accent[i-1] == "o" or accent[i-1] == "N"):
                        continue
                
            if accent[i] == "b" or accent[i] == "c"  or accent[i] == "d" or accent[i] == "g" or accent[i] == "h" or accent[i] == "k" or accent[i] == "m" or accent[i] == "n" or accent[i] == "p" or accent[i] == "r" or accent[i] == "s" or accent[i] == "t":
                if i != len(accent):
                    if accent[i+1] == "[" or accent[i+1] == "["  or accent[i+1] == "#":
                        if i+1 == len(accent):
                            continue
                        if accent[i+2] == "y" or accent[i+2] == "h" or accent[i+2] == "l" or accent[i+2] == "s":
                            #i += 2
                            for j in range(i+1, i+3):
                                skip.append(j)
                            text.append(accent[i]+accent[i+2])
                            continue
                    if accent[i+1] == "y" or accent[i+1] == "h" or accent[i+1] == "l" or accent[i+1] == "s":
                        #i += 1
                        for j in range(i+1, i+2):
                            skip.append(j)
                        text.append(accent[i]+accent[i+1])
                        continue
            text.append(accent[i])
        text.append('$')

        #for i in range(0, len(accent)):
            #text.append(accent[i])
        #text.append('$')
        print(text)
        print(type(text))
        wav, sr = self.engine.tts(text, tqdm=None, spk_id=self.xvector, phoneme=True)
        wav = delay(wav, 2)
        stream_out.write(wav.astype(dtype).tobytes())
    
    def accentRepairePosition(self):
        print("アクセント位置")
        if self.accent_flag == True:
            return

        try:
            self.graph_Items.clear()
            #phoneme = pp_symbols(pyopenjtalk.extract_fullcontext(self.textbox.text()))
            phoneme = ['^']
            accent = self.accent_repaire_edit.toPlainText()

            print(accent)
            skip = []

            for i in range(0, len(accent)):
                if i in skip:
                    continue
                #"by","ch","cl","dy","gy","hy","ky","my","ny","py","ry","sh","ts","ty",
                if  accent[i] == "[" or accent[i] == "["  or accent[i] == "#":
                    if i != 0:
                        if not (accent[i-1] == "a" or accent[i-1] == "i" or accent[i-1] == "u" or accent[i-1] == "e" or accent[i-1] == "o" or accent[i-1] == "N"):
                            continue
                        
                
                if accent[i] == "b" or accent[i] == "c"  or accent[i] == "d" or accent[i] == "g" or accent[i] == "h" or accent[i] == "k" or accent[i] == "m" or accent[i] == "n" or accent[i] == "p" or accent[i] == "r" or accent[i] == "s" or accent[i] == "t":
                    if i != len(accent):
                        if accent[i+1] == "[" or accent[i+1] == "["  or accent[i+1] == "#":
                            if i+1 == len(accent):
                                continue
                            if accent[i+2] == "y" or accent[i+2] == "h" or accent[i+2] == "l" or accent[i+2] == "s":
                                #i += 2
                                for j in range(i+1, i+3):
                                    skip.append(j)
                                print(accent[i]+accent[i+2])
                                print(i)
                                phoneme.append(accent[i]+accent[i+2])
                                continue
                        if accent[i+1] == "y" or accent[i+1] == "h" or accent[i+1] == "l" or accent[i+1] == "s":
                            #i += 1
                            for j in range(i+1, i+2):
                                skip.append(j)
                            print(accent[i]+accent[i+1])
                            print(i)
                            phoneme.append(accent[i]+accent[i+1])
                            continue
                phoneme.append(accent[i])
                print(phoneme)
                print(i)
            phoneme.append('$')
            #print("フルコン",pyopenjtalk.extract_fullcontext(self.textbox.text()))
            print("音素",phoneme)

            #self.scroll_accent.setRange(0, len(katakana) - 1)
            #self.scroll_accent.setValue(0)
            kata = "".join(phoneme)
            x_list = []
            y_list = []
            x_labels = []
            accent_position = 0
            for i in range(1, len(phoneme) - 1):
                if phoneme[i] == "#":
                    if 1 not in y_list:
                        y_list[0] = 1
                    self.graph_Items.addItem(pg.PlotCurveItem(
                    x_list,
                    y_list
                    ))
                    x_list = []
                    y_list = []
                    continue
                if phoneme[i] == "[": 
                    accent_position = 1
                    continue
                if phoneme[i] == "]":
                    accent_position = 0
                    continue

                x_list.append(i)
                y_list.append(accent_position)
                x_labels.append((i, phoneme[i]))
            if 1 not in y_list:
                 y_list[0] = 1

            self.graph_Items.addItem(pg.PlotCurveItem(
                x_list,
                y_list
            ))
            
            self.ax_bottom.setTicks([x_labels])
        except Exception as e:
            print(e)
            print("エラーだよい")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()