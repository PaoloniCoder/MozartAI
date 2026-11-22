() => {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const init = () => {
        const hero = document.querySelector(".studio-hero");
        const tips = document.querySelector(".studio-tips");
        const footer = document.querySelector(".studio-footer");
        if (!hero || !tips || !footer) return false;

        if (window.Swiper && !tips.swiper) {
            new window.Swiper(tips, {
                slidesPerView: 1,
                spaceBetween: 10,
                speed: reduceMotion ? 0 : 300,
                breakpoints: { 760: { slidesPerView: 2 } },
                observer: true,
                observeParents: true,
                navigation: {
                    nextEl: tips.querySelector(".studio-tip-next"),
                    prevEl: tips.querySelector(".studio-tip-prev"),
                },
                a11y: {
                    prevSlideMessage: "Consiglio precedente",
                    nextSlideMessage: "Consiglio successivo",
                },
            });
        }

        if (!reduceMotion && window.gsap) {
            window.gsap.from(hero.querySelectorAll(".studio-score i"), {
                y: 22, opacity: 0, stagger: 0.09, duration: 0.65, ease: "power2.out",
            });
            if (window.ScrollTrigger) {
                window.gsap.registerPlugin(window.ScrollTrigger);
                window.gsap.to(hero.querySelector(".studio-score"), {
                    y: -30,
                    ease: "none",
                    scrollTrigger: { trigger: hero, start: "top top", end: "bottom top", scrub: true },
                });
            }
        }

        if (!reduceMotion && window.AOS) {
            footer.dataset.aos = "fade-up";
            window.AOS.init({ once: true, duration: 400, offset: 0 });
        }
        return true;
    };

    const start = () => {
        if (init()) return;
        const observer = new MutationObserver(() => {
            if (init()) observer.disconnect();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    };
    if (document.readyState === "complete") start();
    else window.addEventListener("load", start, { once: true });
}
