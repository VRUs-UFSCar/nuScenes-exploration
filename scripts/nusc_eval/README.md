
# Framework avaliação nuScenes 3D

Para avaliar o desempenho dos modelos que fazem previsões de boxes 3D, é preciso que sejam geradas as detecções/inferências em toda a base de dados de validação da nuScenes. O formato do JSON esperado é informado na [página do desafio de detecção](https://nuscenes.org/object-detection).

Com o código disponível em `eval.py` é possível calculcar as métricas que o modelo obteve no conjunto de validação.

Para calcular as métricas, execute o seguinte comando (dentro do repositório raiz do projeto):

`python scripts/nusc_eval/eval.py [caminho-json] --output_dir [caminho-saida-metricas]`

Neste comando, substitua os placeholders:
- `[caminho-json]`: caminho relativo à raiz do projeto até o arquivo em que está salvo o JSON contendo as detecções do modelo. Recomenda-se colocar esse JSON no seguinte caminho: `outputs/multi_modality/[nome-modelo]/detections_val.json`
- `[caminho-saida-metricas]`: caminho onde serão salvas as métricas calculadas pelo código. Recomenda-se definir o caminho seguindo o padrão `outputs/multi_modality/[nome-modelo]/metrics`

Existem outros argumentos que podem ser usados para a execução do script de avaliação. Para tal, consulte o código `eval.py`