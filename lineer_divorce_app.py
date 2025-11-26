# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 16:23:38 2025

@author: user
"""

# app.py
# İlişki Dinamikleri - Streamlit Arayüzü
# 54 soruluk form -> zeta, wn, kutuplar, sıfırlar, transfer fonksiyonu + yorumlar

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

try:
    import control as ct
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False
    # Kullanıcıya ekranda uyarı vereceğiz.


# =============== MODEL FONKSİYONLARI =============== #

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@st.cache_data
def fit_model_params_from_dataset(excel_path="divorce_data excel.xltx"):
    """
    Dataset üzerinden:
    - Soru bazlı mean/std (z-score için)
    - Grup bazlı min/max (min-max normalize için)
    parametreleri üretir ve bir sözlük döner.
    """
    df = pd.read_excel(excel_path)

    # Q1..Q54
    Qcols = [f"Q{i}" for i in range(1, 55)]
    Q = df[Qcols].values   # (N,54)

    # Soru bazında z-score fit et (mean, std saklayacağız)
    scaler = StandardScaler().fit(Q)
    Q_norm = scaler.transform(Q)

    # Grup tanımları
    groups = {
        "A": list(range(10, 21)),  # Q10-Q20
        "B": list(range(5, 10)),   # Q5-Q9
        "C": list(range(21, 31)),  # Q21-Q30
        "D": list(range(1, 5)),    # Q1-Q4
        "E": list(range(31, 42)),  # Q31-Q41
        "F": list(range(42, 48)),  # Q42-Q47
        "G": list(range(48, 55)),  # Q48-Q54
    }

    # Grup skorları (dataset için)
    group_scores = {}
    for g, qs in groups.items():
        idxs = [i-1 for i in qs]  # 1-based -> 0-based
        group_scores[g] = Q_norm[:, idxs].mean(axis=1)

    A_raw = group_scores["A"]
    B_raw = group_scores["B"]
    C_raw = group_scores["C"]
    D_raw = group_scores["D"]
    E_raw = group_scores["E"]
    F_raw = group_scores["F"]
    G_raw = group_scores["G"]

    # D ve E için min-max (zeta/wn için)
    D_min, D_max = D_raw.min(), D_raw.max()
    E_min, E_max = E_raw.min(), E_raw.max()

    # Diğer gruplar için min-max (reel pole ve zero'lar için)
    A_min, A_max = A_raw.min(), A_raw.max()
    B_min, B_max = B_raw.min(), B_raw.max()
    C_min, C_max = C_raw.min(), C_raw.max()
    F_min, F_max = F_raw.min(), F_raw.max()
    G_min, G_max = G_raw.min(), G_raw.max()
    D2_min, D2_max = D_raw.min(), D_raw.max()  # zero hesabında D için

    params = {
        "Q_mean": scaler.mean_,        # shape: (54,)
        "Q_scale": scaler.scale_,      # shape: (54,)
        "groups": groups,

        "D_min": D_min, "D_max": D_max,
        "E_min": E_min, "E_max": E_max,

        "A_min": A_min, "A_max": A_max,
        "B_min": B_min, "B_max": B_max,
        "C_min": C_min, "C_max": C_max,
        "F_min": F_min, "F_max": F_max,
        "G_min": G_min, "G_max": G_max,
        "D2_min": D2_min, "D2_max": D2_max,
    }

    return params


def evaluate_user_from_answers(answers, params, K=1.0):
    """
    answers: uzunluğu 54 olan liste veya numpy array (Q1..Q54, 0-4 arası)
    params: fit_model_params_from_dataset() çıktısı
    K: DC kazanç (şimdilik 1 bırakılabilir)
    """

    answers = np.asarray(answers, dtype=float)  # (54,)

    # 1) Soru bazında z-score (dataset'in mean/std'si ile)
    Q_mean = params["Q_mean"]
    Q_scale = params["Q_scale"]
    Q_norm_user = (answers - Q_mean) / (Q_scale + 1e-8)

    # 2) Grup skorları
    groups = params["groups"]
    def group_mean(q_idxs):
        idxs = [i-1 for i in q_idxs]  # 1-based -> 0-based
        return Q_norm_user[idxs].mean()

    A = group_mean(groups["A"])
    B = group_mean(groups["B"])
    C = group_mean(groups["C"])
    D = group_mean(groups["D"])
    E = group_mean(groups["E"])
    F = group_mean(groups["F"])
    G = group_mean(groups["G"])

    # 3) D ve E'yi min-max normalize et (dataset parametreleriyle)
    D_min, D_max = params["D_min"], params["D_max"]
    E_min, E_max = params["E_min"], params["E_max"]

    D_n = (D - D_min) / (D_max - D_min + 1e-8)
    E_n = (E - E_min) / (E_max - E_min + 1e-8)

    # 4) zeta ve wn (aynı formüller)
    zeta_min, zeta_max = 0.2, 1.2
    wn_min, wn_max     = 0.4, 2.0
    k_zeta, k_wn       = 4.0, 4.0

    D_diff = D_n - E_n
    E_diff = E_n - D_n

    zeta = zeta_min + (zeta_max - zeta_min) * sigmoid(k_zeta * D_diff)
    wn   = wn_min   + (wn_max   - wn_min)   * sigmoid(k_wn   * E_diff)

    # 5) Reel pole'ler için A,B,F,G min-max normalize (dataset parametreleriyle)
    def minmax_norm(x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8)

    A_n = minmax_norm(A, params["A_min"], params["A_max"])
    B_n = minmax_norm(B, params["B_min"], params["B_max"])
    F_n = minmax_norm(F, params["F_min"], params["F_max"])
    G_n = minmax_norm(G, params["G_min"], params["G_max"])

    FG_n = 0.5*(F_n + G_n)

    tauA0, tauB0, tauG0 = 10.0, 7.0, 12.0
    kA = kB = kG = 1.0

    tau_A = tauA0 / (1.0 + kA * A_n)
    tau_B = tauB0 / (1.0 + kB * B_n)
    tau_G = tauG0 / (1.0 + kG * FG_n)

    p3 = -1.0 / tau_A
    p4 = -1.0 / tau_B
    p5 = -1.0 / tau_G

    # 6) Çekirdek kutuplar (p1, p2)
    disc = 1.0 - zeta**2
    disc_clipped = max(disc, 0.0)

    real_part = -zeta * wn
    imag_part = wn * np.sqrt(disc_clipped)

    p1 = real_part + 1j * imag_part
    p2 = real_part - 1j * imag_part

    # 7) Zero'lar (C ve D'den, yine dataset min-max'i ile normalize)
    C_n = minmax_norm(C, params["C_min"], params["C_max"])
    D2_n = minmax_norm(D, params["D2_min"], params["D2_max"])

    gamma_C0, gamma_D0 = 0.5, 1.5
    alpha_C,  alpha_D  = 0.5, 0.5

    gamma_C = max(gamma_C0 + alpha_C * C_n, 0.1)
    gamma_D = max(gamma_D0 - alpha_D * D2_n, 0.1)

    zC = gamma_C * wn
    zD = gamma_D * wn

    # 8) Polinomları oluştur
    core_den = np.poly1d([1.0, 2*zeta*wn, wn**2])
    den_p3   = np.poly1d([1.0, -p3])
    den_p4   = np.poly1d([1.0, -p4])
    den_p5   = np.poly1d([1.0, -p5])

    trend_den = den_p3 * den_p4 * den_p5

    num_zC = np.poly1d([1.0, zC])
    num_zD = np.poly1d([1.0, zD])

    num = num_zC * num_zD
    den = core_den * trend_den

    num_coeffs = K * np.array(num.coeffs, dtype=float)
    den_coeffs = np.array(den.coeffs,     dtype=float)

    return {
        "A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G,
        "zeta": float(zeta),
        "wn": float(wn),
        "poles": [p1, p2, p3, p4, p5],
        "zeros": [-zC, -zD],
        "zC": float(zC),
        "zD": float(zD),
        "num": num_coeffs,
        "den": den_coeffs
    }


def interpret_results_streamlit(res):
    """Hesaplanan parametrelere göre yorumları Streamlit'e yazar."""
    zeta = res["zeta"]
    wn   = res["wn"]
    poles = res["poles"]
    zeros = res["zeros"]

    st.subheader("Model Parametre Yorumları")

    # --- ZETA YORUMU ---
    st.markdown(f"**Sönüm oranı (ζ):** `{zeta:.3f}`")
    if zeta < 0.4:
        st.write("• **Düşük sönüm:** İlişkide duygusal salınımlar yüksek, çatışma döngüsel hale gelebilir.")
    elif 0.4 <= zeta <= 0.8:
        st.write("• **Orta düzey sönüm:** Çatışma var ama sistem genelde kendi dengesini bulabiliyor.")
    else:
        st.write("• **Yüksek sönüm:** Çatışmalar genelde hızlı söner; ilişki oldukça stabil veya bastırılmış olabilir.")

    # --- WN YORUMU ---
    st.markdown(f"**Doğal frekans (ωₙ):** `{wn:.3f}`")
    if wn < 0.8:
        st.write("• **Düşük frekans:** İlişkide değişimler yavaş, tepkiler daha geç ortaya çıkıyor.")
    elif 0.8 <= wn <= 1.4:
        st.write("• **Orta frekans:** Ne çok hızlı ne çok yavaş; tartışma ve toparlanma süreleri dengeli.")
    else:
        st.write("• **Yüksek frekans:** Duygusal tepkiler ve iniş-çıkışlar daha hızlı yaşanıyor.")

    # --- POLE YORUMU ---
    st.markdown("**Kutup (pole) konumları:**")
    for i, p in enumerate(poles, start=1):
        st.write(f"• p{i} = {p.real:.3f} {'+' if p.imag>=0 else '-'} {abs(p.imag):.3f}j")

    p1, p2 = poles[0], poles[1]
    if abs(p1.imag) > 1e-3:
        st.write("• Çekirdek kutuplar **kompleks**: İlişkide belirgin bir salınım (iniş-çıkış döngüleri) yapısı var.")
    else:
        st.write("• Çekirdek kutuplar **reel**: Çatışma dinamikleri daha az salınımlı, daha monoton bir sönüm görülüyor.")

    real_parts = [p.real for p in poles]
    slow_modes = [rp for rp in real_parts if rp > -0.15]  # orijine yakın
    fast_modes = [rp for rp in real_parts if rp < -0.5]

    if slow_modes:
        st.write("• Bazı kutuplar orijine yakın: İlişkide yavaş değişen, zor çözülen kalıplar mevcut.")
    if fast_modes:
        st.write("• Bazı kutuplar oldukça solda: Bazı tepkiler hızlı veriliyor veya bazı sorunlar çabuk çözülüyor.")

    # --- ZERO YORUMU ---
    st.markdown("**Sıfır (zero) konumları:**")
    for i, z in enumerate(zeros, start=1):
        st.write(f"• z{i} = {z:.3f}")

    z1, z2 = zeros
    if max(z1, z2) > -0.5:
        st.write("• Bazı zero'lar orijine yakın: C ve D gruplarına ait kısa vadeli davranışların anlık etkisi yüksek.")
    else:
        st.write("• Zero'lar daha solda: C ve D'nin etkisi daha çok sistemin genel formunu şekillendiriyor, anlık tepkiyi az etkiliyor.")

    st.info("Bu yorumlar model-tabanlıdır; klinik tanı yerine dinamik bir bakış açısı sunar.")


# =============== SORU METİNLERİ (Q1–Q54) =============== #

QUESTIONS = [
"Aramızdaki tartışma kötüleştiğinde, birimiz özür dilerse tartışma biter.",
"Zaman zaman zor olsa bile, farklılıklarımızı görmezden gelebileceğimizi biliyorum.",
"Gerektiğinde eşimle tartışmalarımızı baştan ele alıp düzeltebiliriz.",
"Eşimle tartıştığımda, iletişim kurmaya çalışmak sonunda işe yarar.",
"Eşimle geçirdiğim zaman ikimiz için özeldir.",
"Evde eş olarak birlikte zaman geçiremiyoruz.",
"Evde aile olmaktan çok aynı ortamı paylaşan iki yabancı gibiyiz.",
"Eşimle tatillerimizden keyif alırım.",
"Eşimle seyahat etmekten keyif alırım.",
"Eşimle çoğu hedefimiz ortaktır.",
"İleride bir gün geriye baktığımda eşimle uyum içinde olduğumuzu göreceğimi düşünüyorum.",
"Eşimle kişisel özgürlük konusunda benzer değerlere sahibiz.",
"Eşimle benzer bir eğlenme anlayışına sahibiz.",
"İnsanlara (çocuklar, arkadaşlar vb.) yönelik çoğu hedefimiz aynıdır.",
"Eşimle hayallerimiz benzerdir ve birbirini tamamlar.",
"Aşkın nasıl olması gerektiği konusunda eşimle uyumluyuz.",
"Eşimle hayatımızda mutlu olmak konusunda aynı görüşleri paylaşırız.",
"Evliliğin nasıl olması gerektiği hakkında eşimle benzer fikirlerimiz var.",
"Evlilikte rollerin nasıl olması gerektiği hakkında eşimle benzer düşüncelere sahibiz.",
"Güven konusunda eşimle benzer değerlere sahibiz.",
"Eşimin nelerden hoşlandığını tam olarak biliyorum.",
"Eşim hasta olduğunda nasıl ilgi görmek istediğini bilirim.",
"Eşimin en sevdiği yemeği bilirim.",
"Eşimin hayatında hangi streslerle karşı karşıya olduğunu söyleyebilirim.",
"Eşimin iç dünyası hakkında bilgi sahibiyim.",
"Eşimin temel kaygılarını bilirim.",
"Eşimin şu anda yaşadığı stres kaynaklarını bilirim.",
"Eşimin umutlarını ve dileklerini bilirim.",
"Eşimi çok iyi tanırım.",
"Eşimin arkadaşlarını ve sosyal ilişkilerini bilirim.",
"Eşimle tartışırken kendimi saldırgan hissederim.",
"Eşimle tartışırken genelde \"sen her zaman...\" veya \"sen hiç...\" gibi ifadeler kullanırım.",
"Tartışmalarımız sırasında eşimin kişiliğiyle ilgili olumsuz ifadeler kullanabilirim.",
"Tartışmalarımızda kırıcı ifadeler kullanabilirim.",
"Tartışmalarımız sırasında eşime hakaret edebilirim.",
"Tartışırken aşağılayıcı olabilirim.",
"Eşimle tartışmalarım sakin geçmez.",
"Eşimin bir konuyu açma şeklinden nefret ederim.",
"Tartışmalarımız çoğu zaman aniden ortaya çıkar.",
"Ne olduğunu anlamadan tartışmaya başlarız.",
"Eşimle bir şey konuşurken sakinliğim bir anda bozulur.",
"Eşimle tartıştığımda dışarı çıkarım ve tek kelime etmem.",
"Ortamı biraz yatıştırmak için çoğunlukla sessiz kalırım.",
"Bazen bir süreliğine evden uzaklaşmanın benim için iyi olduğunu düşünürüm.",
"Eşimle tartışmaktansa sessiz kalmayı tercih ederim.",
"Tartışmada haklı olsam bile eşimi incitmemek için sessiz kalırım.",
"Eşimle tartışırken öfkemi kontrol edememekten korktuğum için sessiz kalırım.",
"Tartışmalarımızda kendimi haklı hissederim.",
"Suçlandığım şeylerle benim bir ilgim yok.",
"Aslında suçlandığım konularda hatalı olan ben değilim.",
"Evdeki sorunlarda hata bende değildir.",
"Eşimin yetersizliğini ona söylemekten çekinmem.",
"Tartıştığımızda eşime yetersizliğini hatırlatırım.",
"Eşimin beceriksizliğini söylemekten korkmam."
]


# =============== STREAMLIT ARAYÜZÜ =============== #

def main():
    st.set_page_config(page_title="İlişki Dinamikleri Modeli", layout="wide")

    st.title("İlişki Dinamikleri – Zeta, ωₙ, Pole–Zero ve Transfer Fonksiyonu")
    st.write(
        "Bu arayüz, 54 maddelik ilişki ölçeğini kullanarak "
        "ilişkinizi ikinci dereceden çekirdek + ek modlardan oluşan **5 kutuplu** bir sistem olarak modelliyor."
    )

    if not HAS_CONTROL:
        st.warning(
            "`python-control` paketi yüklü değil. Transfer fonksiyonu grafikleri (step, pzmap) "
            "için aşağıdakini kurman gerekiyor:\n\n"
            "`pip install control`"
        )

    # Dataset parametrelerini yükle
    with st.spinner("Dataset parametreleri yükleniyor..."):
        params = fit_model_params_from_dataset("divorce_data excel.xltx")

    st.success("Dataset parametreleri yüklendi.")

    st.markdown("### 54 Soru – Lütfen her maddeyi 0–4 arasında cevaplayın")
    st.caption("0 = Kesinlikle katılmıyorum, 4 = Tamamen katılıyorum (veya ölçeğin tanımına göre).")

    answers = []

    # Grupları bloklayarak göstermek için indeks aralıklarını kullanabiliriz:
    # D: Q1-4, B: Q5-9, A: Q10-20, C: Q21-30, E: Q31-41, F: Q42-47, G: Q48-54
    group_blocks = [
        ("D Grubu (Onarım / Tartışma Çözümü)", range(0, 4)),
        ("B Grubu (Ortak Zaman / Yakınlık)", range(4, 9)),
        ("A Grubu (Uyum ve Ortak Değerler)", range(9, 20)),
        ("C Grubu (Eşi Tanıma / Duygusal Farkındalık)", range(20, 30)),
        ("E Grubu (Saldırgan Tartışma Davranışları)", range(30, 41)),
        ("F Grubu (Sessizlik / Geri Çekilme)", range(41, 47)),
        ("G Grubu (Suçlama / Yetersizlik Atfetme)", range(47, 54)),
    ]

    # Soruları expanders ile gruplu gösterelim
    for group_name, idx_range in group_blocks:
        with st.expander(group_name, expanded=True):
            for i in idx_range:
                q_text = QUESTIONS[i]
                val = st.slider(
                    f"Q{i+1}: {q_text}",
                    min_value=0,
                    max_value=4,
                    value=2,
                    step=1,
                    key=f"q{i+1}"
                )
                answers.append(val)

    # Hesapla butonu
    if st.button("Modeli Hesapla"):
        if len(answers) != 54:
            st.error("Tüm 54 soruyu cevaplaman gerekiyor.")
            return

        with st.spinner("Model hesaplanıyor..."):
            res = evaluate_user_from_answers(answers, params)

        st.success("Model hesaplandı.")

        # Temel parametreler
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sönüm Oranı (ζ)", f"{res['zeta']:.3f}")
        with col2:
            st.metric("Doğal Frekans (ωₙ)", f"{res['wn']:.3f}")

        # Kutuplar ve sıfırlar tablosu
        st.subheader("Kutup ve Sıfır Değerleri")

        poles_list = []
        for i, p in enumerate(res["poles"], start=1):
            poles_list.append({
                "Kutup": f"p{i}",
                "Re": p.real,
                "Im": p.imag
            })
        poles_df = pd.DataFrame(poles_list)

        zeros_list = []
        for i, z in enumerate(res["zeros"], start=1):
            zeros_list.append({
                "Sıfır": f"z{i}",
                "Değer": z
            })
        zeros_df = pd.DataFrame(zeros_list)

        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(poles_df)
        with c2:
            st.dataframe(zeros_df)

        # Transfer fonksiyon katsayıları
        st.subheader("Transfer Fonksiyonu Katsayıları")
        st.write("**Pay (num):**", res["num"])
        st.write("**Payda (den):**", res["den"])

        # Yorumlar
        interpret_results_streamlit(res)

        # Eğer control paketi varsa step ve pzmap grafikleri
        if HAS_CONTROL:
            G = ct.TransferFunction(res["num"], res["den"])

            st.subheader("Step Response")
            t, y = ct.step_response(G)
            fig_step, ax_step = plt.subplots()
            ax_step.plot(t, y)
            ax_step.set_xlabel("Zaman")
            ax_step.set_ylabel("İlişki Tepkisi")
            ax_step.grid(True)
            st.pyplot(fig_step)

            st.subheader("Pole-Zero Haritası")
            fig_pz, ax_pz = plt.subplots()
            ct.pzmap(G, plot=True, grid=True, ax=ax_pz)
            st.pyplot(fig_pz)

        else:
            st.warning("`python-control` yüklü olmadığı için grafikler çizilemiyor. "
                       "İstersen `pip install control` ile kurup tekrar çalıştırabilirsin.")


if __name__ == "__main__":
    main()
