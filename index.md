---
layout: default
title: "FreqExit: Enabling Early-Exit Inference for Visual Autoregressive Models via Frequency-Aware Guidance"
permalink: "/"
---

<!-- ===== Fonts & minimal style (works on GitHub Pages) ===== -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root{
    --text:#1f2328; --muted:#57606a; --accent:#1f6feb; --pill:#2f363d; --gold:#d4a72c;
    --maxw:1100px;
  }
  body{
    font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji";
  }
  .wrap{
    max-width:var(--maxw);
    margin:0 auto;
    padding:24px 16px 48px;
  }
  .title{
    font-size:44px;
    font-weight:800;
    line-height:1.15;
    text-align:center;
    color:var(--text);
    margin:18px 0 8px;
  }
  .venue{
    font-size:22px;
    font-weight:700;
    color:#d12;
    text-align:center;
    margin:4px 0 18px;
  }
  .authors,.affils{
    text-align:center;
    color:var(--muted);
  }
  .authors a{
    color:#0969da;
    text-decoration:none;
  }
  .authors a:hover{
    text-decoration:underline;
  }
  sup{font-size:.75em}
  .badges{
    text-align:center;
    margin:26px 0 20px;
  }
  .badge{
    display:inline-block;
    margin:6px 8px;
    padding:10px 14px;
    border-radius:14px;
    background:var(--pill);
    color:#fff;
    font-weight:700;
    letter-spacing:.3px;
    text-decoration:none;
    box-shadow:0 2px 6px rgba(0,0,0,.12);
  }
  .badge.blue{ background:var(--accent);}
  .badge.gold{ background:var(--gold); color:#1f2328;}
  .center{ text-align:center;}
  .section{
    max-width: var(--maxw);
    margin: 28px auto 0;
    padding: 0 4px;
  }
  .logos{
    display:flex;
    justify-content:center;
    align-items:center;
    gap:32px;
    margin:20px 0 28px;
  }
  .logos img{
    height:64px;
    width:auto;
    object-fit:contain;
    max-width:220px;
  }

 
  img.hero{
    max-width:420px;
    width:100%;
    border-radius:10px;
    box-shadow:0 8px 24px rgba(0,0,0,.08);
  }

  
  img.method-fig{
    max-width:900px;
    width:100%;
    border-radius:10px;
    box-shadow:0 8px 24px rgba(0,0,0,.08);
  }

  
  .carousel{
    position:relative;
    overflow:hidden;
    max-width:var(--maxw);
    margin:12px auto 0;
    border-radius:10px;
    box-shadow:0 8px 24px rgba(0,0,0,.06);
  }
  .carousel-track{
    display:flex;
    transition:transform .6s ease;
  }
  
  .carousel-slide{
    flex:0 0 100%;
    display:flex;
    justify-content:center;
    align-items:center;
    padding:8px 0;
  }
  .carousel-slide img{
    display:block;
    max-width:100%;  
    height:auto;
  }

  
  .result-img-small{
    max-width:80%;   
  }
  .result-img-native{
    max-width:100%;  
  }

  .carousel-dots{
    text-align:center;
    margin-top:8px;
  }
  .carousel-dot{
    display:inline-block;
    width:8px;
    height:8px;
    border-radius:999px;
    background:#d0d7de;
    margin:0 4px;
    cursor:pointer;
  }
  .carousel-dot.active{
    background:var(--accent);
  }
</style>

<div class="wrap">

  <h1 class="title">FreqExit: Enabling Early-Exit Inference for Visual Autoregressive Models via Frequency-Aware Guidance</h1>
  <div class="venue">NeurIPS 2025</div>

  <div class="authors">
    <a>Ying Li</a><sup>1</sup>,
    <a href="https://openreview.net/profile?id=~chengfei_lv1" target="_blank">Chengfei Lv</a><sup>2</sup>,
    <a href="https://huanwang.tech/" target="_blank">Huan Wang</a><sup>1</sup>
  </div>

  <div class="affils">
    <sup>1</sup> Westlake University &nbsp;&nbsp;·&nbsp;&nbsp;
    <sup>2</sup> Alibaba Group
  </div>

  <div class="center" style="margin-top:6px;">
    <em>*Corresponding author: wanghuan [at] westlake [dot] edu [dot] cn</em>
  </div>

  <div class="logos">
    <img src="Figures/westlake.png" alt="Westlake University logo">
    <img src="Figures/Alibaba-group.png" alt="Alibaba Group logo">
  </div>

  <div class="badges">
    <a class="badge blue" href="https://github.com/NeuraLiying/FreqExit">Code</a>
    <a class="badge gold" href="https://opensource.org/license/apache-2-0">Apache&nbsp;2.0</a>
  </div>

  <div class="center">
    <img class="hero" src="Figures/supplementary_generation.png" alt="FreqExit overview figure">
    <p><em>FreqExit bridges step-wise generation and early-exit acceleration, achieving up to <strong>2×</strong> speedup with negligible quality degradation.</em></p>
  </div>

  <!-- ===== Sections written in pure HTML so they always render correctly ===== -->
  <div class="section">
    <h2>Abstract</h2>
    <p>
      FreqExit is a dynamic inference framework for Visual AutoRegressive (VAR) models, which decode from coarse
      structures to fine details. Existing methods fail on VAR due to the absence of semantic stability and smooth
      representation transitions. FreqExit addresses this by recognizing that high-frequency details essential to
      visual quality tend to emerge in later decoding stages. On ImageNet 256×256, FreqExit achieves up to
      <b>2×</b> speedup with only minor degradation, and delivers <b>1.3×</b> acceleration without perceptible quality
      loss. This enables runtime-adaptive acceleration within a unified model, offering a favorable trade-off between
      efficiency and fidelity for practical and flexible deployment.
    </p>
  </div>

  <div class="section">
    <h2>Overview of our FreqExit method</h2>
    <div class="center">
      <img class="method-fig" src="Figures/method.png" alt="Overview of the FreqExit method">
    </div>
  </div>

  <div class="section">
    <h2>Main Results</h2>
    <div class="carousel" data-interval="5000">
      <div class="carousel-track">
        <div class="carousel-slide">
          <img src="Figures/exp_1.png" alt="Main results figure 1" class="result-img-small">
        </div>
        <div class="carousel-slide">
          <img src="Figures/exp_2.png" alt="Main results figure 2" class="result-img-native">
        </div>
      </div>
    </div>
    <div class="carousel-dots">
      <span class="carousel-dot active"></span>
      <span class="carousel-dot"></span>
    </div>
  </div>

  <div class="section">
    <h2>More Visualizations</h2>
    <div class="carousel" data-interval="5000">
      <div class="carousel-track">
        <div class="carousel-slide">
          <img src="Figures/supplementary_generation.png" alt="Additional FreqExit generation results">
        </div>
        <div class="carousel-slide">
          <img src="Figures/supplementary_inpaint.png" alt="Additional FreqExit inpainting results">
        </div>
      </div>
    </div>
    <div class="carousel-dots">
      <span class="carousel-dot active"></span>
      <span class="carousel-dot"></span>
    </div>
  </div>

  <div class="section">
    <h2>Acknowledgement</h2>
    <p>
      This work builds upon the foundations of prior open-source efforts,
      including <a href="https://github.com/FoundationVision/VAR">VAR</a>,
      <a href="https://github.com/czg1225/CoDe">CoDe</a>, and
      <a href="https://github.com/facebookresearch/LayerSkip">LayerSkip</a>.
      We sincerely thank the authors for their excellent contributions to the research community.
    </p>
  </div>

  <div class="section">
    <h2>BibTeX</h2>
    <pre><code class="language-bibtex">@inproceedings{li2025freqexit,
  title={FreqExit: Enabling Early-Exit Inference for Visual Autoregressive Models via Frequency-Aware Guidance},
  author={Li, Ying and Lv, Chengfei and Wang, Huan},
  booktitle={NeurIPS},
  year={2025}
}</code></pre>
  </div>

</div> <!-- /wrap -->

<script>
document.addEventListener('DOMContentLoaded', function () {
  const carousels = document.querySelectorAll('.carousel');

  carousels.forEach(function (carousel) {
    const track = carousel.querySelector('.carousel-track');
    const slides = Array.from(track.children);
    const dotsContainer = carousel.parentElement.querySelector('.carousel-dots');
    const dots = dotsContainer ? dotsContainer.querySelectorAll('.carousel-dot') : [];
    let currentIndex = 0;
    const interval = parseInt(carousel.dataset.interval || '5000', 10);

    function showSlide(i) {
      if (!slides.length) return;
      currentIndex = (i + slides.length) % slides.length;
      track.style.transform = 'translateX(' + (-currentIndex * 100) + '%)';
      dots.forEach(function (dot, k) {
        dot.classList.toggle('active', k === currentIndex);
      });
    }

    dots.forEach(function (dot, i) {
      dot.addEventListener('click', function () {
        showSlide(i);
      });
    });

    showSlide(0);
    setInterval(function () {
      showSlide(currentIndex + 1);
    }, interval);
  });
});
</script>
