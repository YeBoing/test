import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI小眼-MVP", layout="wide", page_icon="👁️")


def init_state():
    defaults = {
        "active_page": "工作台",
        "patient_id": None,
        "patient_code": None,
        "screening_id": None,
        "last_result": None,
        "step_uploaded": False,
        "step_analyzed": False,
        "screening_step": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def post_json(url: str, payload: dict):
    resp = requests.post(url, json=payload, timeout=20)
    try:
        body = resp.json()
    except Exception:
        body = {"detail": "服务返回格式异常"}
    return resp.status_code, body


def go_page(page: str):
    st.session_state["active_page"] = page


def risk_label(level: str | None):
    level = (level or "").lower()
    mapping = {
        "none": ("无风险提示", "🟢"),
        "low": ("低风险", "🟡"),
        "medium": ("中风险", "🟠"),
        "high": ("高风险", "🔴"),
    }
    return mapping.get(level, ("待分析", "⚪"))


def risk_color(level: str | None):
    level = (level or "").lower()
    mapping = {
        "none": "#16a34a",
        "low": "#ca8a04",
        "medium": "#f97316",
        "high": "#dc2626",
    }
    return mapping.get(level, "#64748b")


def risk_badge(title: str, level: str | None):
    text, _ = risk_label(level)
    color = risk_color(level)
    st.markdown(
        f"""
        <div style="padding:14px 16px;border-radius:10px;background:#fff;border:1px solid #e2e8f0;">
            <div style="font-size:14px;color:#334155;">{title}</div>
            <div style="margin-top:8px;display:inline-block;background:{color};color:#fff;padding:6px 10px;border-radius:999px;font-size:13px;font-weight:600;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_advice_sections(dr_level: str | None, htn_level: str | None, recommendation_text: str | None):
    levels = [dr_level or "", htn_level or ""]
    if "high" in levels:
        timeline = "建议 1 周内前往眼科/专科医院进一步检查。"
        management = "注意血压、血糖日常监测，避免熬夜，遵医嘱管理基础慢病。"
        followup = "系统建议随访：高优先级。"
    elif "medium" in levels:
        timeline = "建议 1-3 个月内复查，并进行专科咨询。"
        management = "建议规律作息、饮食管理，并记录血压/血糖变化。"
        followup = "系统建议随访：中优先级。"
    elif "low" in levels:
        timeline = "建议 3-6 个月复查。"
        management = "保持当前健康管理，减少高盐高糖饮食。"
        followup = "系统建议随访：常规提醒。"
    else:
        timeline = "建议年度常规眼底复查。"
        management = "保持健康生活方式，定期体检。"
        followup = "系统建议随访：可选。"

    return {
        "结果解读": recommendation_text or "当前为辅助筛查结果，请结合临床进一步判断。",
        "就医时限建议": timeline,
        "健康管理建议": management,
        "复查与随访": followup,
    }


def show_step_progress(current_step: int):
    current_step = max(1, min(3, int(current_step)))
    progress_map = {1: 0.34, 2: 0.67, 3: 1.0}
    st.progress(progress_map[current_step], text=f"当前进度：{current_step}/3")

    labels = [
        ("1/3 创建患者", current_step > 1, current_step == 1),
        ("2/3 上传眼底图", current_step > 2, current_step == 2),
        ("3/3 查看评估结果", current_step == 3, current_step == 3),
    ]
    cols = st.columns(3)
    for col, (label, done, active) in zip(cols, labels):
        color = "#16a34a" if done else ("#2563eb" if active else "#94a3b8")
        bg = "#dcfce7" if done else ("#dbeafe" if active else "#f1f5f9")
        col.markdown(
            f"""
            <div style="padding:8px 10px;border-radius:8px;background:{bg};color:{color};font-weight:600;text-align:center;">
                {'✓ ' if done else ('● ' if active else '')}{label}
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_result_panel(result: dict):
    dr_level = result.get("dr_risk_level")
    htn_level = result.get("htn_risk_level")
    dr_text, dr_icon = risk_label(dr_level)
    htn_text, htn_icon = risk_label(htn_level)

    st.subheader("筛查结果")
    m1, m2, m3 = st.columns(3)
    m1.metric("糖网风险", f"{dr_icon} {dr_text}")
    m2.metric("高血压相关风险", f"{htn_icon} {htn_text}")
    m3.metric("建议随访", "需要" if result.get("followup_needed") == "Y" else "常规")

    b1, b2 = st.columns(2)
    with b1:
        risk_badge("糖网风险等级", dr_level)
    with b2:
        risk_badge("高血压相关风险等级", htn_level)

    st.markdown("### 医疗建议")
    sections = build_advice_sections(dr_level, htn_level, result.get("recommendation_text"))
    for title, content in sections.items():
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(content)

    st.caption("本系统仅用于辅助筛查演示，不构成诊断结论，请以专科医生意见为准。")


def render_workbench(backend: str):
    st.subheader("工作台")
    st.write("快速开始：先创建患者，再创建筛查任务，点击“上传并开始评估”。")
    a, b, c = st.columns(3)
    a.metric("当前患者ID", st.session_state.get("patient_id") or "-")
    b.metric("当前任务ID", st.session_state.get("screening_id") or "-")
    c.metric("状态", "已就绪" if st.session_state.get("screening_id") else "待创建")

    if st.button("去新建筛查", type="primary"):
        st.session_state["screening_step"] = 1
        go_page("新建筛查")
        st.rerun()


def render_new_screening(backend: str):
    st.subheader("新建筛查")
    current_step = int(st.session_state.get("screening_step", 1))
    show_step_progress(current_step)

    if current_step == 1:
        st.markdown("### 第一步：创建患者")
        with st.form("patient_form", clear_on_submit=False):
            name = st.text_input("姓名")
            gender = st.selectbox("性别", ["male", "female"])
            age = st.number_input("年龄", min_value=1, max_value=120, value=45)
            phone_masked = st.text_input("手机号(脱敏)", value="138****8888")
            submit_patient = st.form_submit_button("保存并进入下一步")

        if submit_patient:
            code, res = post_json(
                f"{backend}/patients",
                {
                    "name": name,
                    "gender": gender,
                    "age": int(age),
                    "phone_masked": phone_masked,
                },
            )
            if code == 200:
                st.session_state["patient_id"] = res["id"]
                st.session_state["patient_code"] = res["patient_code"]
                st.session_state["step_uploaded"] = False
                st.session_state["step_analyzed"] = False
                st.session_state["screening_step"] = 2
                st.success(f"患者已保存：{res['name']}（编号：{res['patient_code']}）")
                st.rerun()
            else:
                st.error(res.get("detail", "患者保存失败"))

        return

    if current_step == 2:
        st.markdown("### 第二步：上传眼底图并自动评估")
        st.caption(f"当前患者ID：{st.session_state.get('patient_id') or '-'}")

        checkup_date = st.text_input("体检日期", value="2026-07-26")
        left_image = st.file_uploader("左眼图片", type=["jpg", "jpeg", "png"], key="left")
        right_image = st.file_uploader("右眼图片", type=["jpg", "jpeg", "png"], key="right")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("上一步", use_container_width=True):
                st.session_state["screening_step"] = 1
                st.rerun()
        with c2:
            start = st.button("上传并开始AI评估", type="primary", use_container_width=True)

        if start:
            patient_id = st.session_state.get("patient_id")
            if not patient_id:
                st.error("请先创建患者")
                st.session_state["screening_step"] = 1
                st.rerun()
                return

            files = {}
            if left_image is not None:
                files["left_image"] = (left_image.name, BytesIO(left_image.read()), left_image.type)
            if right_image is not None:
                files["right_image"] = (right_image.name, BytesIO(right_image.read()), right_image.type)

            if not files:
                st.warning("请至少上传一张眼底图")
                return

            # 第二步内自动创建筛查任务
            code, created = post_json(
                f"{backend}/screenings",
                {"patient_id": int(patient_id), "checkup_date": checkup_date},
            )
            if code != 200:
                st.error(created.get("detail", "创建筛查任务失败"))
                return

            sid = int(created["id"])
            st.session_state["screening_id"] = sid

            upload_resp = requests.post(f"{backend}/screenings/{sid}/images", files=files, timeout=30)
            if upload_resp.status_code != 200:
                try:
                    detail = upload_resp.json().get("detail", "上传失败")
                except Exception:
                    detail = "上传失败"
                st.error(detail)
                return

            st.session_state["step_uploaded"] = True

            analyze_resp = requests.post(f"{backend}/screenings/{sid}/analyze", timeout=30)
            if analyze_resp.status_code == 200:
                st.session_state["last_result"] = analyze_resp.json()
                st.session_state["step_analyzed"] = True
                st.session_state["screening_step"] = 3
                go_page("结果中心")
                st.rerun()
            else:
                try:
                    detail = analyze_resp.json().get("detail", "AI评估失败")
                except Exception:
                    detail = "AI评估失败"
                st.error(detail)

        return


def render_result_center(backend: str):
    st.subheader("结果中心")
    show_step_progress(3)

    result = st.session_state.get("last_result")
    if result:
        show_result_panel(result)

    st.markdown("### 查询指定任务结果")
    qid = st.number_input("任务ID", min_value=1, value=int(st.session_state.get("screening_id") or 1), key="qid")
    if st.button("加载结果", use_container_width=True):
        resp = requests.get(f"{backend}/screenings/{int(qid)}/result", timeout=20)
        if resp.status_code == 200:
            result = resp.json()
            st.session_state["last_result"] = result
            show_result_panel(result)
        else:
            try:
                detail = resp.json().get("detail", "获取结果失败")
            except Exception:
                detail = "获取结果失败"
            st.error(detail)


def render_history(backend: str):
    st.subheader("历史记录")
    if st.button("刷新记录", type="primary"):
        resp = requests.get(f"{backend}/screenings", timeout=20)
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                table = pd.DataFrame(rows)
                table = table.rename(
                    columns={
                        "screening_id": "任务ID",
                        "patient_code": "患者编号",
                        "patient_name": "姓名",
                        "checkup_date": "体检日期",
                        "status": "状态",
                        "qc_status": "质控",
                        "dr_risk_level": "糖网风险",
                        "htn_risk_level": "高血压相关风险",
                        "followup_needed": "需随访",
                    }
                )
                st.dataframe(table, use_container_width=True, hide_index=True)
            else:
                st.info("暂无记录")
        else:
            st.error("获取记录失败")


init_state()

st.markdown(
    """
    <style>
    .stApp { background-color: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI小眼｜体检中心眼底筛查系统")
st.caption("业务流程：登记 → 上传 → 自动AI评估 → 风险分层 → 建议动作")

with st.sidebar:
    st.subheader("系统导航")
    backend = st.text_input("后端地址", BACKEND_URL)
    pages = ["工作台", "新建筛查", "结果中心", "历史记录"]
    current_page = st.session_state.get("active_page", "工作台")
    if current_page not in pages:
        current_page = "工作台"
    nav = st.radio("页面", pages, index=pages.index(current_page))
    st.session_state["active_page"] = nav

page = st.session_state["active_page"]

if page == "工作台":
    render_workbench(backend)
elif page == "新建筛查":
    render_new_screening(backend)
elif page == "结果中心":
    render_result_center(backend)
else:
    render_history(backend)
