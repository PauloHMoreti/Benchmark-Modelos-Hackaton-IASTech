# Benchmark de detectores de objetos para P&IDs

Compara três arquiteturas de detecção de objetos no mesmo dataset, para
decidir qual usar no reconhecimento de equipamentos/símbolos em P&IDs
(o PaddleOCR fica de fora — ele cuida do texto/tags separadamente):

| Modelo | Papel no benchmark |
|---|---|
| **RT-DETR** | Candidato principal: melhor equilíbrio esperado entre precisão e velocidade |
| **Faster R-CNN** | Referência de precisão (mais lento, mais conservador) |
| **RetinaNet** | Avalia o comportamento com o desbalanceamento de classes (Focal Loss) |

Pipeline do projeto:

```
P&ID → [ detector: RT-DETR / Faster R-CNN / RetinaNet ] → equipamentos
     → [ PaddleOCR ]                                     → tags
                              ↓
                      pós-processamento → Excel
```

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `benchmark_pid_models.py` | Script principal: treina (fine-tuning) e avalia os 3 modelos, gera a comparação |
| `convert_annotations.py` | Converte anotações exportadas do Make Sense AI (YOLO `.txt` ou Pascal VOC `.xml`) para COCO |
| `requirements.txt` | Dependências Python |

## Formato de dados esperado

O script trabalha com anotações no formato **COCO (json)** — é o único formato
que as três arquiteturas conseguem consumir sem conversão escondida. Organize
o dataset assim:

```
dataset/
    images/
        train/   *.jpg ou *.png
        val/     *.jpg ou *.png
    annotations/
        train.json
        val.json
```

Se o Make Sense AI exportar em YOLO ou Pascal VOC, use o conversor **antes**
de rodar o benchmark:

```bash
# formato YOLO (um .txt por imagem + classes.txt)
python convert_annotations.py yolo \
    --images-dir dataset_bruto/images/train \
    --labels-dir dataset_bruto/labels/train \
    --classes-file dataset_bruto/classes.txt \
    --output dataset/annotations/train.json

# formato Pascal VOC (um .xml por imagem)
python convert_annotations.py voc \
    --images-dir dataset_bruto/images/train \
    --annotations-dir dataset_bruto/annotations_voc/train \
    --output dataset/annotations/train.json
```

Rode uma vez para `train` e uma vez para `val`. As imagens também precisam
estar copiadas em `dataset/images/train` e `dataset/images/val`.

> **Atenção (YOLO):** o id de cada classe é a posição da linha em
> `classes.txt`. Use o **mesmo** `classes.txt`, na mesma ordem, para
> converter treino e validação — senão os ids ficam inconsistentes entre
> os dois splits.

## Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

Isso instala `torch`, `torchvision`, `transformers`, `scipy` (usado pelo
matcher húngaro do RT-DETR), `pycocotools`, `pandas`, `matplotlib`, `Pillow`
e `tqdm`.

A primeira execução baixa automaticamente os pesos pré-treinados em COCO de
cada arquitetura (do PyTorch Hub e do Hugging Face) — é necessário ter
internet nesse momento. Os arquivos ficam em cache local; execuções
seguintes não baixam de novo.

## Como rodar

### 1. Teste de fumaça (recomendado antes de tudo)

Roda o pipeline inteiro com um dataset sintético (retângulos coloridos),
só para confirmar que a instalação e o código funcionam de ponta a ponta
antes de usar o dataset real:

```bash
python benchmark_pid_models.py --smoke-test --epochs 1
```

### 2. Benchmark com o dataset real

```bash
python benchmark_pid_models.py \
    --data-dir ./dataset \
    --epochs 15 \
    --batch-size 4 \
    --output-dir ./resultados_benchmark
```

### Principais argumentos

| Argumento | Padrão | Descrição |
|---|---|---|
| `--data-dir` | `./dataset` | Pasta com `images/` e `annotations/` em COCO |
| `--output-dir` | `./resultados_benchmark` | Onde salvar checkpoints, tabelas e gráfico |
| `--epochs` | `15` | Número de épocas de fine-tuning (igual para os 3 modelos) |
| `--batch-size` | `4` | Tamanho do batch |
| `--lr` | `0.005` | Learning rate do SGD (Faster R-CNN / RetinaNet — o RT-DETR usa AdamW com lr fixo) |
| `--models` | os 3 | Rodar só um subconjunto, ex.: `--models faster_rcnn retinanet` |
| `--rtdetr-checkpoint` | `PekingU/rtdetr_r50vd_coco_o365` | Checkpoint do Hugging Face usado como ponto de partida |
| `--smoke-test` | desligado | Usa dataset sintético em vez de `--data-dir` |

Rode `python benchmark_pid_models.py --help` para ver tudo.

## O que o script faz, por modelo

1. Carrega pesos pré-treinados em COCO;
2. Faz fine-tuning no dataset de vocês pelo mesmo número de épocas
   (otimizador recomendado por arquitetura: SGD para Faster R-CNN/RetinaNet,
   AdamW para RT-DETR — forçar os mesmos hiperparâmetros nos três seria
   injusto, já que Transformers e CNNs treinam de formas diferentes);
3. Avalia no conjunto de validação: mAP@[.5:.95], mAP@.5, mAP@.75, AP por
   tamanho de objeto (small/medium/large), AP por classe, latência
   (ms/imagem) e FPS;
4. Salva o checkpoint treinado.

## Saídas geradas (`--output-dir`)

| Arquivo | Conteúdo |
|---|---|
| `resultados_resumo.csv` | Tabela comparativa (uma linha por modelo) |
| `resultados_completos.json` | Mesma tabela + AP por classe (`per_class_AP@[.5:.95]`) |
| `comparacao.png` | Gráfico de barras: mAP@[.5:.95] x FPS |
| `<modelo>.pt` | Pesos do modelo já treinado no dataset de vocês |

## Como interpretar os resultados

- **mAP@[.5:.95] vs mAP@.5**: o `@.5` é mais tolerante a pequenos
  desalinhamentos de bbox — em símbolos pequenos de P&ID, costuma ser mais
  relevante para decisão do que o `@[.5:.95]`, que exige sobreposição quase
  perfeita.
- **AP por classe** (não só o mAP agregado): o dataset tem classes
  desbalanceadas (ex.: válvula ≫ britador). Um mAP bom pode esconder uma
  classe rara com AP próximo de zero — é aqui que se espera ver o RetinaNet
  se destacar (ou não) por causa da Focal Loss.
- **AP_small**: símbolos de P&ID tendem a cair na faixa "small" da escala
  COCO — esse número importa mais que o AP geral para avaliar objetos
  pequenos e próximos.
- **Velocidade/FPS**: pesa pouco se o processamento for uma imagem por vez,
  mas ainda afeta o tempo total de processar o lote de P&IDs.
- **Poucas imagens de validação = métricas instáveis**: com datasets
  pequenos, uma diferença de 2-3 pontos de mAP pode ser ruído. Rodar mais
  de uma seed antes de decidir ajuda a não escolher pelo acaso.
- **RT-DETR precisa de mais épocas para convergir** que CNNs com FPN. Com
  poucas épocas ele pode parecer artificialmente pior sem que a arquitetura
  seja de fato inferior para o problema.
- **Inspecionar erros visualmente**: além das métricas, vale abrir algumas
  imagens de validação e ver que tipo de erro cada modelo comete — símbolo
  pequeno não detectado, classe confundida com outra visualmente parecida,
  ou bbox certa com classe errada.
- **Impacto no pipeline completo**: uma bbox "quase certa" pode ser
  suficiente para o mAP, mas cortar mal a região usada pelo PaddleOCR. Vale
  checar o resultado do pipeline completo (equipamento + tag) nos casos que
  cada modelo erra por pouco.

## Problemas conhecidos / troubleshooting

- **`ImportError: RTDetrHungarianMatcher requires the scipy library`**:
  falta `scipy` no ambiente — já está no `requirements.txt`; rode
  `pip install scipy` se o erro aparecer.
- **`[transformers] You passed num_labels=3 which is incompatible...` e
  `MISMATCH: Reinit due to size mismatch`**: mensagens esperadas. O
  checkpoint do RT-DETR vem pré-treinado em 80 classes; o script troca a
  cabeça de classificação para as classes de vocês. Não é erro.
  
- **RetinaNet sem nenhuma detecção nas primeiras épocas
  (`[AVISO] Nenhuma detecção gerada`)**: também esperado. O torchvision
  inicializa a cabeça de classificação do RetinaNet com viés bem negativo
  (o truque da Focal Loss), então logo no início ele pode não gerar nenhuma
  detecção acima do threshold de confiança. Tende a se resolver com mais
  épocas de treino.
- **Download de pesos falha / trava**: os pesos do Faster R-CNN/RetinaNet
  vêm do PyTorch Hub (`download.pytorch.org`) e os do RT-DETR do Hugging
  Face (`huggingface.co`) — é preciso ter esses domínios acessíveis na
  primeira execução.