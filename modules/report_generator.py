import pandas as pd
import io
import os
import tempfile
from PIL import Image
from fpdf import FPDF
import streamlit as st

# Import your charting functions
from .charts import (
    plot_expense_by_category,
    plot_need_vs_want,
    plot_top_expenses,
    plot_monthly_trends
)

# -------------------------
#  PDF CLASS DEFINITION
# -------------------------
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)

    def header(self):
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 14, 'FinSight - Financial Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 16)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(8)

    def chapter_body(self, body_md):
        # Corrected width to avoid text cutoff
        usable_width = self.w - self.l_margin - self.r_margin
        for line in body_md.split('\n'):
            line = line.strip()
            if line.startswith('## '):
                self.set_font('DejaVu', 'B', 12)
                self.multi_cell(usable_width, 7, txt=line[3:], align='J')
                self.ln(2)
            elif line.startswith('* '):
                self.set_font('DejaVu', '', 10)
                self.multi_cell(usable_width, 6, txt=f'  •  {line[2:]}', align='J')
                self.ln(1)
            elif line:
                self.set_font('DejaVu', '', 10)
                self.multi_cell(usable_width, 6.5, txt=line, align='J')
                self.ln(0.5)
        self.ln()

    # -------------------------
    #  HELPER: Save figures as temp PNG
    # -------------------------
    def save_fig_to_png(self, fig, prefix="chart"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=prefix)
        tmp.close()
        path = tmp.name
        try:
            if hasattr(fig, "to_image"):
                img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
                with open(path, "wb") as f:
                    f.write(img_bytes)
            elif hasattr(fig, "write_image"):
                fig.write_image(path, format="png")
            else:
                fig.savefig(path, bbox_inches="tight", dpi=150)

            im = Image.open(path)
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                bg.save(path, "PNG")
            else:
                im.save(path, "PNG")
            return path
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            print(f"[ERROR] Could not save figure: {e}")
            return None

    # -------------------------
    #  FIXED CHART GRID (LANDSCAPE OPTIMIZED)
    # -------------------------
    def add_chart_grid(self, fig1, fig2, title1, title2, gap_mm=10):
        y_before = self.get_y()
        if y_before + 90 > self.h - self.b_margin:
            self.add_page()
            y_before = self.get_y()

        usable_width = self.w - self.l_margin - self.r_margin
        col_w = (usable_width - gap_mm) / 2.0
        left_x = self.l_margin
        right_x = self.l_margin + col_w + gap_mm

        img1_path = self.save_fig_to_png(fig1, prefix="chart_left") if fig1 else None
        img2_path = self.save_fig_to_png(fig2, prefix="chart_right") if fig2 else None

        img1_h_mm = img2_h_mm = 0
        if img1_path:
            with Image.open(img1_path) as im:
                w_px, h_px = im.size
                img1_h_mm = (h_px / w_px) * col_w
        if img2_path:
            with Image.open(img2_path) as im:
                w_px, h_px = im.size
                img2_h_mm = (h_px / w_px) * col_w

        self.set_xy(left_x, y_before)
        self.set_font('DejaVu', 'B', 11)
        self.multi_cell(col_w, 5, title1, 0, 'L')
        left_y = self.get_y()

        self.set_xy(right_x, y_before)
        self.multi_cell(col_w, 5, title2, 0, 'L')
        right_y = self.get_y()

        top_y = max(left_y, right_y) + 3
        placeholder_h = 45
        used_h = max(img1_h_mm or placeholder_h, img2_h_mm or placeholder_h)

        if img1_path:
            self.image(img1_path, x=left_x, y=top_y, w=col_w)
        else:
            self.rect(left_x, top_y, col_w, placeholder_h)
            self.set_xy(left_x + 2, top_y + 2)
            self.cell(col_w - 4, 5, 'Chart unavailable', 0, 1, 'L')

        if img2_path:
            self.image(img2_path, x=right_x, y=top_y, w=col_w)
        else:
            self.rect(right_x, top_y, col_w, placeholder_h)
            self.set_xy(right_x + 2, top_y + 2)
            self.cell(col_w - 4, 5, 'Chart unavailable', 0, 1, 'L')

        self.set_y(top_y + used_h + 8)

        for p in (img1_path, img2_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# -------------------------
#  HELPER FUNCTION
# -------------------------
def _wrap_text(pdf, text, max_width_mm):
    words = str(text).split()
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        test = cur + (" " if cur else "") + w
        if pdf.get_string_width(test) <= max_width_mm:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# -------------------------
#  REPORT CREATION FUNCTION
# -------------------------
def create_report(df: pd.DataFrame, metrics: dict, ai_summary: str) -> bytes:
    pdf = PDF(orientation='L', unit='mm', format='A4')
    pdf.set_margins(left=20, top=15, right=25)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # -------- Summary Section --------
    pdf.chapter_title('📈Your Financial Summary')
    pdf.chapter_body(ai_summary)

    # -------- Visual Analysis --------
    pdf.add_page()
    pdf.chapter_title('📊 Visual Analysis')

    with st.spinner("Generating charts for PDF..."):
        fig_cat = plot_expense_by_category(df)
        fig_need = plot_need_vs_want(df)
        fig_top = plot_top_expenses(df)
        fig_trends = plot_monthly_trends(df)

        if fig_cat and fig_need:
            pdf.add_chart_grid(fig_cat, fig_need, "Expenses by Category", "Need vs. Want Spending")

        pdf.ln(10)

        if fig_top and fig_trends:
            pdf.add_chart_grid(fig_top, fig_trends, "Top 10 Largest Expenses", "Monthly Trends")

    # -------- Transaction Table --------
    pdf.add_page()
    pdf.chapter_title('📋 Transaction Data')

    headers = ['Date', 'Description', 'Category', 'Debit', 'Credit']
    col_widths = [25, 100, 40, 40, 40]
    alignments = ['L', 'L', 'L', 'R', 'R']

    pdf.set_fill_color(230, 230, 230)
    pdf.set_font('DejaVu', 'B', 8)
    for header, width in zip(headers, col_widths):
        pdf.cell(width, 8, header, 1, 0, 'C', fill=True)
    pdf.ln()

    pdf.set_font('DejaVu', '', 7)
    fill = False
    line_height = 4.5

    df_report = df.copy()
    df_report['Date'] = pd.to_datetime(df_report['Date']).dt.strftime('%Y-%m-%d')
    df_report['Debit'] = df_report['Debit'].fillna(0.0)
    df_report['Credit'] = df_report['Credit'].fillna(0.0)

    for _, row in df_report[headers].iterrows():
        desc_lines = _wrap_text(pdf, row['Description'], col_widths[1] - 1)
        date_lines = _wrap_text(pdf, row['Date'], col_widths[0] - 1)
        cat_lines = _wrap_text(pdf, row['Category'], col_widths[2] - 1)
        debit_lines = _wrap_text(pdf, f"{row['Debit']:.2f}", col_widths[3] - 1)
        credit_lines = _wrap_text(pdf, f"{row['Credit']:.2f}", col_widths[4] - 1)

        max_lines = max(len(desc_lines), len(date_lines), len(cat_lines), len(debit_lines), len(credit_lines))
        row_h = line_height * max_lines

        if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            pdf.set_font('DejaVu', 'B', 8)
            for header, width in zip(headers, col_widths):
                pdf.cell(width, 8, header, 1, 0, 'C', fill=True)
            pdf.ln()
            pdf.set_font('DejaVu', '', 7)

        x_start = pdf.get_x()
        y_start = pdf.get_y()

        pdf.cell(col_widths[0], row_h, date_lines[0] if date_lines else '', 'LR', 0, 'L', fill)
        desc_x = x_start + col_widths[0]
        pdf.set_xy(desc_x, y_start)
        pdf.multi_cell(col_widths[1], line_height, "\n".join(desc_lines), border='LR', align='L', fill=fill)
        pdf.set_xy(desc_x + col_widths[1], y_start)

        pdf.cell(col_widths[2], row_h, cat_lines[0] if cat_lines else '', 'LR', 0, 'L', fill)
        pdf.cell(col_widths[3], row_h, debit_lines[0] if debit_lines else '0.00', 'LR', 0, 'R', fill)
        pdf.cell(col_widths[4], row_h, credit_lines[0] if credit_lines else '0.00', 'LR', 0, 'R', fill)

        pdf.ln(row_h)
        fill = not fill

    pdf.cell(sum(col_widths), 0, '', 'T')
    return bytes(pdf.output(dest='S'))