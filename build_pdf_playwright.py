import os
import sys
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import pypdf

def is_page_blank(page):
    text = page.extract_text() or ""
    if len(text.strip()) > 0:
        return False
    try:
        if len(page.images) > 0:
            return False
    except Exception:
        pass
    return True

def main():
    html_path = "d:/First Year/First Year 3rd sem/Maths/project/New project/dissertation.html"
    staged_html_path = "d:/First Year/First Year 3rd sem/Maths/project/New project/dissertation_staged.html"
    temp_pdf_path = "d:/First Year/First Year 3rd sem/Maths/project/New project/temp_dissertation.pdf"
    temp_cleaned_path = "d:/First Year/First Year 3rd sem/Maths/project/New project/temp_cleaned.pdf"
    temp_final_path = "d:/First Year/First Year 3rd sem/Maths/project/New project/temp_final.pdf"
    final_pdf_path = "d:/First Year/First Year 3rd sem/Maths/project/New project/Comprehensive_Advertising_Sales_Analysis_Report.pdf"

    print("--- [Pass 1] Preparing HTML for initial PDF generation ---")
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # 1. Inject custom styles for TOC dot leaders, cover page, and fix overlapping content
    style_tag = soup.new_tag("style")
    style_tag.string = """
    /* Playwright/Chromium TOC styles */
    .toc li, .lof-list li, .lot-list li {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 6pt;
    }
    .toc li a, .lof-list li a, .lot-list li a {
      text-decoration: none;
      color: #1a1a2e;
      border-bottom: none !important;
      display: inline;
    }
    .toc li a::after, .lof-list li a::after, .lot-list li a::after {
      content: none !important;
    }
    .toc li .dots, .lof-list li .dots, .lot-list li .dots {
      flex-grow: 1;
      border-bottom: 1px dotted #CBD5E1;
      margin: 0 8pt;
    }
    .toc li .page-num, .lof-list li .page-num, .lot-list li .page-num {
      font-family: 'Inter', sans-serif;
      font-size: 10pt;
      color: #64748B;
    }
    /* FIX COVER PAGE OVERLAP: keep in normal document flow */
    .cover {
      width: 210mm;
      height: 240mm;
      box-sizing: border-box;
      position: relative;
      page-break-after: always;
      margin: 0 auto;
      overflow: hidden;
      background: #FFFFFF;
    }
    /* FIX BLANK PAGES & OVERLAPPING CONTENT */
    .chapter-divider {
      height: auto !important;
      min-height: 160mm !important;
      max-height: 220mm !important;
      box-sizing: border-box !important;
      page-break-before: always !important;
      page-break-after: always !important;
      margin: 0 !important;
      padding: 40mm 20mm !important;
    }
    .front-matter-page {
      height: auto !important;
      min-height: 100mm !important;
      max-height: none !important; /* Allow TOC/LOF/LOT to flow naturally across multiple pages without overlapping */
      box-sizing: border-box !important;
      page-break-after: always !important;
      overflow: visible !important;
    }
    """
    soup.head.append(style_tag)

    # 1b. Completely Redesign Cover Page for Majestic Apple/Adobe Academic Report Aesthetics
    cover_div = soup.find("div", class_="cover")
    if cover_div:
        cover_div["style"] = "display: flex; flex-direction: column; justify-content: space-between; height: 240mm; box-sizing: border-box; padding: 10mm 15mm; margin: 0 auto; page-break-after: always; background: #FFFFFF;"
        cover_div.clear()
        
        cover_content = """
        <div class="cover-top" style="text-align: center; margin-top: 10px;">
            <div style="font-family: 'Inter', sans-serif; font-size: 14pt; font-weight: 700; color: #64748B; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 40px;">
                Department of Mathematics
            </div>
            <h1 style="font-family: 'Georgia', serif; font-size: 32pt; font-weight: 800; color: #0F172A; text-align: center; line-height: 1.25; margin: 30px 0 40px 0;">
                Comprehensive Advertising Sales Analysis Using Multiple Linear Regression and Artificial Neural Networks
            </h1>
            <div style="font-family: 'Georgia', serif; font-size: 15pt; font-style: italic; color: #334155; text-align: center; max-width: 85%; margin: 0 auto; line-height: 1.6;">
                A Final Year Dissertation Submitted in Partial Fulfillment of the Requirements for the Course Mathematics for Data Science & Analytics (MDSA)
            </div>
        </div>
        
        <div class="cover-middle" style="margin: 30px auto; width: 100%; text-align: center; font-family: 'Inter', sans-serif;">
            <div style="margin-bottom: 30px;">
                <span style="font-size: 11pt; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 2px; display: block; margin-bottom: 8px;">Submitted By</span>
                <span style="font-size: 14pt; font-weight: 700; color: #1E293B; display: block;">J Sasank <span style="font-weight:400; color:#64748B;">(2510030267)</span> &nbsp;|&nbsp; R Sai Karthik <span style="font-weight:400; color:#64748B;">(2510030295)</span> &nbsp;|&nbsp; B Shriyan <span style="font-weight:400; color:#64748B;">(2510030057)</span></span>
            </div>
            <div style="margin-bottom: 30px;">
                <span style="font-size: 11pt; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 2px; display: block; margin-bottom: 8px;">Under The Supervision Of</span>
                <span style="font-size: 16pt; font-weight: 700; color: #1E293B; display: block;">Dr. M. Nagesh</span>
            </div>
            <div style="margin-bottom: 20px;">
                <span style="font-size: 11pt; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 2px; display: block; margin-bottom: 8px;">Institution</span>
                <span style="font-size: 16pt; font-weight: 700; color: #1E293B; display: block;">KL University — Department of Mathematics</span>
            </div>
        </div>
        
        <div class="cover-bottom" style="text-align: center; font-family: 'Inter', sans-serif; font-size: 14pt; font-weight: 700; color: #0F172A; letter-spacing: 4px; text-transform: uppercase; border-top: 2px solid #E2E8F0; padding-top: 25px; margin-bottom: 10px;">
            Academic Year 2025 – 2026
        </div>
        """
        cover_div.append(BeautifulSoup(cover_content, "html.parser"))

    # 2. Add Figure 16 into the HTML body right after Figure 15 if not already present
    if not soup.find(id="fig-16"):
        fig15 = soup.find(id="fig-15")
        if fig15:
            fig16_tag = soup.new_tag("figure", id="fig-16")
            img_tag = soup.new_tag("img", src="images/fig16_residual_comparison.png", alt="Residual Distribution Comparison")
            figcaption = soup.new_tag("figcaption", **{"data-label": "Figure 16"})
            figcaption.append(BeautifulSoup("<strong>Residual Distribution Comparison — MLR vs. ANN.</strong> Left: Overlay of residual histograms with Kernel Density Estimation (KDE) curves for both Multiple Linear Regression and Artificial Neural Network models. Right: Boxplot comparison of residual spreads. <em>Source:</em> Computed from test set residuals (n=40). <em>Observation:</em> Both models exhibit residuals centred near zero. The ANN residuals demonstrate a slightly narrower interquartile range. <em>Technical Interpretation:</em> The residual analysis confirms that both models satisfy the normality assumption, with ANN displaying marginally lower error variance. <em>Business Interpretation:</em> Errors in sales forecasting are symmetrically distributed and bounded, minimizing the risk of severe budget misallocation. <em>Recommendation:</em> Incorporate the residual standard error into risk assessment models when establishing financial reserves for marketing campaigns.", "html.parser"))
            fig16_tag.append(img_tag)
            fig16_tag.append(figcaption)
            fig15.insert_after(fig16_tag)

    # 3. Add Sections 9.1, 9.2, and 9.3 (Future Work) to Table of Contents if not present
    toc_ul = soup.find("ul", class_="toc")
    if toc_ul:
        if not toc_ul.find("a", href="#sec9-1"):
            ch9_li = toc_ul.find("a", href="#chapter9")
            if ch9_li and ch9_li.parent:
                sec_items = [
                    ("#sec9-1", "9.1 Conclusion"),
                    ("#sec9-2", "9.2 Limitations"),
                    ("#sec9-3", "9.3 Future Work")
                ]
                ch9_section = soup.find(id="chapter9")
                if ch9_section:
                    h2s = ch9_section.find_all("h2")
                    if len(h2s) >= 3:
                        h2s[0]["id"] = "sec9-1"
                        h2s[1]["id"] = "sec9-2"
                        h2s[2]["id"] = "sec9-3"

                curr_li = ch9_li.parent
                for href, text in sec_items:
                    new_li = soup.new_tag("li")
                    new_a = soup.new_tag("a", href=href, class_="toc-section")
                    new_a.string = text
                    new_li.append(new_a)
                    curr_li.insert_after(new_li)
                    curr_li = new_li

    # 4. Modify TOC, LOF, LOT items to include dots and page-num spans, and collect target IDs
    toc_targets = []
    target_titles = {}
    for ul_class in ["toc", "lof-list", "lot-list"]:
        ul = soup.find(["ul", "ol"], class_=ul_class)
        if ul:
            for li in ul.find_all("li"):
                a = li.find("a")
                if a and a.has_attr("href") and a["href"].startswith("#"):
                    target_id = a["href"][1:] # remove #
                    toc_targets.append(target_id)
                    target_titles[target_id] = a.text.strip()
                    dots = soup.new_tag("span", class_="dots")
                    page_span = soup.new_tag("span", class_="page-num", id=f"toc-page-{target_id}")
                    page_span.string = "1" # Placeholder for Pass 1
                    li.append(dots)
                    li.append(page_span)

    # 5. Inject unique invisible anchor tokens right next to target elements in the document
    for target_id in toc_targets:
        elem = soup.find(id=target_id)
        if elem:
            token = soup.new_tag("span", style="font-size: 1pt; color: #FFFFFF; opacity: 0.01;")
            token.string = f"[P-{target_id}]"
            elem.insert(0, token)

    # 6. Inject KaTeX auto-render execution script at the end of body to guarantee perfect math rendering
    script_tag = soup.new_tag("script")
    script_tag.string = "document.addEventListener('DOMContentLoaded', function() { if (window.renderMathInElement) { renderMathInElement(document.body, { delimiters: [ {left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false} ] }); } });"
    if soup.body:
        soup.body.append(script_tag)

    # Save staged HTML
    with open(staged_html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

    print("--- [Pass 1] Launching Playwright to generate temporary PDF ---")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.goto(f"file:///{staged_html_path}")
        page.wait_for_timeout(3000)
        page.pdf(
            path=temp_pdf_path,
            prefer_css_page_size=True,
            display_header_footer=False
        )
        browser.close()

    print("--- [Pass 2] Filtering blank pages from temporary PDF ---")
    reader = pypdf.PdfReader(temp_pdf_path)
    writer = pypdf.PdfWriter()
    
    removed_pages = []
    for i, page in enumerate(reader.pages):
        if is_page_blank(page):
            removed_pages.append(i + 1)
        else:
            writer.add_page(page)
            
    print(f"Blank pages detected and removed in Pass 1: {removed_pages}")
    with open(temp_cleaned_path, "wb") as f:
        writer.write(f)

    print("--- [Pass 2] Analysing cleaned temporary PDF for exact page numbers ---")
    reader_cleaned = pypdf.PdfReader(temp_cleaned_path)
    page_mapping = {}

    for i, page in enumerate(reader_cleaned.pages):
        text = page.extract_text()
        clean_text = "".join(text.split())
        for target_id in toc_targets:
            clean_token = f"[P-{target_id}]"
            if clean_token in clean_text:
                page_mapping[target_id] = i + 1
            else:
                clean_title = "".join(target_titles[target_id].split())
                if clean_title in clean_text:
                    if target_id in ["declaration", "certificate", "acknowledgement", "abstract"]:
                        if target_id not in page_mapping:
                            page_mapping[target_id] = i + 1
                    else:
                        page_mapping[target_id] = i + 1

    print("Mapped Page Numbers after blank page removal:", page_mapping)

    print("--- [Pass 2] Updating HTML with mapped page numbers ---")
    for target_id in toc_targets:
        page_span = soup.find(id=f"toc-page-{target_id}")
        if page_span and target_id in page_mapping:
            page_span.string = str(page_mapping[target_id])

    # Save final staged HTML
    with open(staged_html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

    print("--- [Pass 3] Launching Playwright to generate raw final PDF ---")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.goto(f"file:///{staged_html_path}")
        page.wait_for_timeout(3000)
        page.pdf(
            path=temp_final_path,
            prefer_css_page_size=True,
            display_header_footer=False
        )
        browser.close()

    print("--- [Pass 3] Final blank page pruning for premium output ---")
    final_reader = pypdf.PdfReader(temp_final_path)
    final_writer = pypdf.PdfWriter()
    
    final_removed = []
    for i, page in enumerate(final_reader.pages):
        if is_page_blank(page):
            final_removed.append(i + 1)
        else:
            final_writer.add_page(page)
            
    print(f"Blank pages pruned from final output: {final_removed}")
    with open(final_pdf_path, "wb") as f:
        final_writer.write(f)

    print(f"--- Successfully generated pristine premium PDF at: {final_pdf_path} ---")

if __name__ == "__main__":
    main()
