"""Збирає train/composition_study_colab.ipynb з experiments/shapes.py і model.py.

Запуск: python experiments/build_notebook.py
"""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHAPES_SRC = (ROOT / 'experiments' / 'shapes.py').read_text(encoding='utf-8')
MODEL_SRC = (ROOT / 'experiments' / 'model.py').read_text(encoding='utf-8')
OUT = ROOT / 'train' / 'composition_study_colab.ipynb'


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip()}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': textwrap.dedent(text).strip()}


cells = [
    md('''
        # Склад синтетичного набору і чутливість детектора дефектів

        Серія навчань для статті: 9 складів навчального набору × 3 сіди = 27 моделей
        MobileNetV2, оцінка кожної на спільних тестових наборах.

        **Як запустити**

        1. *Середовище виконання → Змінити тип середовища виконання → GPU (T4).*
        2. *Середовище виконання → Виконати все.* Дозвольте доступ до Google Диска:
           результати пишуться в `MyDrive/defect_study`, тож якщо сесія обірветься,
           просто запустіть усе знову - готові моделі пропускаються.
        3. Близько години. В кінці завантажиться `defect_study_results.zip` - покладіть
           його вміст у `article/results/` репозиторію.

        Для швидкої перевірки, що все працює, поставте `QUICK = True` (кілька хвилин).
    '''),
    code('''
        import os

        try:
            from google.colab import drive, files
            drive.mount('/content/drive')
            ROOT = '/content/drive/MyDrive/defect_study'
        except ImportError:
            files = None
            ROOT = os.path.abspath('defect_study')

        for sub in ('scores', 'models'):
            os.makedirs(f'{ROOT}/{sub}', exist_ok=True)
        print('результати:', ROOT)
    '''),
    md('Генератор силуетів і модель - копії `experiments/shapes.py` і `experiments/model.py` з репозиторію.'),
    code('%%writefile shapes.py\n' + SHAPES_SRC),
    code('%%writefile model.py\n' + MODEL_SRC),
    code('''
        import json
        import time

        import keras
        import numpy as np
        import pandas as pd
        import tensorflow as tf

        import shapes as S
        from model import build_model

        print('TensorFlow', tf.__version__, '| Keras', keras.__version__,
              '| GPU:', tf.config.list_physical_devices('GPU'))

        QUICK = False

        COMPOSITIONS = {
            'L':   {'types': ['line']},
            'B':   {'types': ['bump']},
            'D':   {'types': ['dent']},
            'M':   {'types': ['line', 'blob']},
            'S':   {'types': ['line', 'bump', 'blob']},
            'A':   {'types': list(S.DEFECTS)},
            'T05': {'tolerance': 0.05},
            'T10': {'tolerance': 0.10},
            'T15': {'tolerance': 0.15},
        }
        SEEDS = [0, 1, 2]
        N_SAMPLES = 6000
        EPOCHS_HEAD, EPOCHS_TUNE = 20, 8
        BATCH = 64
        SUITE = dict(per_shape_clean=200, per_cell=120, per_eps=600)
        N_CALIB = 1200

        if QUICK:
            ROOT += '_quick'
            for sub in ('scores', 'models'):
                os.makedirs(f'{ROOT}/{sub}', exist_ok=True)
            COMPOSITIONS = {k: COMPOSITIONS[k] for k in ('S', 'T10')}
            SEEDS, N_SAMPLES, EPOCHS_HEAD, EPOCHS_TUNE = [0], 120, 1, 1
            SUITE, N_CALIB = dict(per_shape_clean=4, per_cell=6, per_eps=6), 24

        with open(f'{ROOT}/config.json', 'w') as fh:
            json.dump({'compositions': COMPOSITIONS, 'seeds': SEEDS, 'n_samples': N_SAMPLES,
                       'epochs': [EPOCHS_HEAD, EPOCHS_TUNE], 'train_ranges': S.TRAIN_RANGES,
                       'grid_h': S.GRID_H, 'blob_h': S.BLOB_H, 'grid_eps': S.GRID_EPS,
                       'suite': SUITE, 'n_calib': N_CALIB}, fh, indent=1)
    '''),
    md('''
        ## Спільні тестові набори

        Генеруються один раз з окремим сідом і кешуються на Диску:
        ідеальні фігури, сітка «тип дефекту × розмір h», нерівні фігури (eps)
        та окремий набір ідеальних фігур для калібрування порога на FPR = 1 %.
    '''),
    code('''
        suite_path = f'{ROOT}/test_suite.npz'
        if os.path.exists(suite_path):
            cached = np.load(suite_path)
            X_test, X_calib = cached['X_test'], cached['X_calib']
            meta = pd.read_csv(f'{ROOT}/test_meta.csv', index_col='id', keep_default_na=False)
        else:
            X_test, meta = S.test_suite(np.random.default_rng(2026), **SUITE)
            X_calib = S.clean_set(N_CALIB, np.random.default_rng(2027))
            meta = pd.DataFrame(meta)
            meta.to_csv(f'{ROOT}/test_meta.csv', index_label='id')
            np.savez_compressed(suite_path, X_test=X_test, X_calib=X_calib)

        print('тест:', X_test.shape, '| калібрування:', X_calib.shape)
        print(meta.groupby(['set', 'defect']).size())
    '''),
    md('## Модель і навчання - як у базовій роботі, рання зупинка за val loss'),
    code('''
        @tf.autograph.experimental.do_not_convert
        def with_rgb(image, label):
            return tf.image.grayscale_to_rgb(tf.cast(image, tf.float32)), label


        @tf.autograph.experimental.do_not_convert
        def only_rgb(image):
            return tf.image.grayscale_to_rgb(tf.cast(image, tf.float32))


        def to_dataset(images, labels=None, shuffle=False, seed=0):
            x = images[..., None]
            if labels is None:
                data = tf.data.Dataset.from_tensor_slices(x).batch(256).map(only_rgb)
            else:
                data = tf.data.Dataset.from_tensor_slices((x, labels))
                if shuffle:
                    data = data.shuffle(4096, seed=seed)
                data = data.batch(BATCH).map(with_rgb)
            return data.prefetch(tf.data.AUTOTUNE)


        def fit(model, base, train_ds, val_ds):
            def stop():
                return keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=4, restore_best_weights=True)

            def compile_with(rate):
                model.compile(optimizer=keras.optimizers.Adam(rate), loss='binary_crossentropy',
                              metrics=['accuracy', keras.metrics.AUC(name='auc')])

            # перемішує tf.data; shuffle=False лише прибирає попередження
            compile_with(1e-3)
            head = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                             callbacks=[stop()], shuffle=False, verbose=0)

            base.trainable = True
            for layer in base.layers[:-30]:
                layer.trainable = False
            compile_with(1e-5)
            tune = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_TUNE,
                             callbacks=[stop()], shuffle=False, verbose=0)
            return head, tune


        def predict(model, images):
            return model.predict(to_dataset(images), verbose=0).ravel().astype(np.float32)


        def auc_of(labels, scores):
            metric = keras.metrics.AUC()
            metric.update_state(labels, scores)
            return float(metric.result())
    '''),
    md('''
        ## Серія навчань

        Для кожного складу і сіду: генерація 6000 зразків (60/20/20), навчання,
        передбачення на власному тесті, на спільних тестових наборах і на
        калібрувальному наборі. Усе зберігається одразу, тож перезапуск продовжує
        з місця обриву.
    '''),
    code('''
        log_path = f'{ROOT}/train_log.csv'
        log = pd.read_csv(log_path).to_dict('records') if os.path.exists(log_path) else []
        done = {(row['comp'], row['seed']) for row in log}

        for ci, (comp, spec) in enumerate(COMPOSITIONS.items()):
            for seed in SEEDS:
                name = f'{comp}_s{seed}'
                if (comp, seed) in done:
                    print('вже є:', name)
                    continue

                started = time.time()
                keras.utils.set_random_seed(1000 * seed + ci)
                rng = np.random.default_rng(10_000 + 100 * ci + seed)
                if 'tolerance' in spec:
                    X, y = S.tolerance_set(spec['tolerance'], N_SAMPLES, rng)
                else:
                    X, y = S.composition_set(spec['types'], N_SAMPLES, rng)

                n_test = n_val = N_SAMPLES // 5
                X_own, y_own = X[:n_test], y[:n_test]
                X_val, y_val = X[n_test:n_test + n_val], y[n_test:n_test + n_val]
                X_train, y_train = X[n_test + n_val:], y[n_test + n_val:]

                model, base = build_model()
                head, tune = fit(model, base,
                                 to_dataset(X_train, y_train, shuffle=True, seed=seed),
                                 to_dataset(X_val, y_val))

                p_own = predict(model, X_own)
                np.savez_compressed(f'{ROOT}/scores/{name}.npz',
                                    test=predict(model, X_test), calib=predict(model, X_calib),
                                    own=p_own, own_labels=y_own)
                # копія без стану оптимізатора: файл удвічі менший
                bare, _ = build_model(weights=None)
                bare.set_weights(model.get_weights())
                bare.save_weights(f'{ROOT}/models/{name}.weights.h5')

                row = {'comp': comp, 'seed': seed,
                       'epochs_head': len(head.history['loss']),
                       'epochs_tune': len(tune.history['loss']),
                       'val_loss': round(float(min(tune.history['val_loss'])), 5),
                       'own_acc': round(float(((p_own > 0.5) == y_own).mean()), 4),
                       'own_auc': round(auc_of(y_own, p_own), 4),
                       'seconds': round(time.time() - started, 1)}
                log.append(row)
                pd.DataFrame(log).to_csv(log_path, index=False)
                print(row, flush=True)

                del model, base, bare
                keras.backend.clear_session()
    '''),
    md('''
        ## Короткий підсумок

        Сід 0, поріг 0.5: частка виявлених дефектів кожного типу при h ≥ 0.10
        і частка хибних тривог на ідеальних фігурах. Повний аналіз - локально.
    '''),
    code('''
        grid = meta['set'] == 'grid'
        big = grid & (meta['h'] >= 0.10)
        rows = []
        for comp in COMPOSITIONS:
            scores = np.load(f'{ROOT}/scores/{comp}_s0.npz')['test']
            row = {'склад': comp, 'FPR ідеальні': (scores[meta['set'] == 'clean'] > 0.5).mean()}
            for defect in S.DEFECTS:
                row[defect] = (scores[big & (meta['defect'] == defect)] > 0.5).mean()
            rows.append(row)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        print()
        print(pd.read_csv(log_path).groupby('comp')[['own_acc', 'own_auc', 'seconds']].mean().round(4))
    '''),
    md('''
        ## Пакування результатів

        В архів ідуть оцінки всіх моделей, метадані тестів, журнал навчання і моделі
        з сідом 0 (для реальних фото). Усе інше лишається на Диску в `defect_study`.
    '''),
    code('''
        import shutil

        pack = '/content/defect_study_results' if files else os.path.abspath('defect_study_results')
        shutil.rmtree(pack, ignore_errors=True)
        os.makedirs(f'{pack}/models')
        for name in ('config.json', 'test_meta.csv', 'train_log.csv'):
            shutil.copy(f'{ROOT}/{name}', pack)
        shutil.copytree(f'{ROOT}/scores', f'{pack}/scores')
        for comp in COMPOSITIONS:
            shutil.copy(f'{ROOT}/models/{comp}_s0.weights.h5', f'{pack}/models')

        archive = shutil.make_archive(pack, 'zip', pack)
        print(archive, round(os.path.getsize(archive) / 2**20, 1), 'МБ')
        if files:
            files.download(archive)
    '''),
]

notebook = {
    'cells': [{**cell, 'source': cell['source'].splitlines(keepends=True)} for cell in cells],
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'name': OUT.name, 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 0,
}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT)
