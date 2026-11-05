"""Microscope ROI & 3D Tissue Plane Generator.
Image/ROI calculations are retained from the supplied microscope ROI application.
Plane mathematics, POS processing and Plotly visualization are retained from the supplied calculator.
"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import calculations as calc
import pos_processor as pos
import visualization as viz
from modules import microscope_core as mc

st.set_page_config(page_title="Microscope ROI & 3D Tissue Plane Generator", page_icon="🔬", layout="wide")
st.markdown("""<style>
/* Theme-aware styling: never hard-code a light surface in dark mode. */
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1500px}
:root {
  --microscope-surface: var(--secondary-background-color);
  --microscope-text: var(--text-color);
  --microscope-border: var(--border-color, rgba(128,128,128,.25));
}
[data-testid="stSidebar"] {
  background-color: var(--microscope-surface) !important;
  color: var(--microscope-text) !important;
}
[data-testid="stSidebar"] * {
  color: inherit;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
  color: var(--microscope-text);
}
[data-testid="stMetric"] {
  background-color: var(--microscope-surface);
  color: var(--microscope-text);
  border: 1px solid var(--microscope-border);
  padding: 12px;
  border-radius: 10px;
}
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  color: var(--microscope-text) !important;
}
/* Keep links and controls legible while respecting Streamlit's active theme. */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  color: var(--microscope-text);
}
</style>""", unsafe_allow_html=True)
st.title("🔬 Microscope ROI & 3D Tissue Plane Generator")
st.caption("Microscope image processing · calibrated ROI generation · stage-coordinate Z-plane calculation · Micro-Manager PositionList validation")

ss=st.session_state
DEFAULTS={"perspective_points":[],"calibration_points":[],"hsv_low_h":125,"hsv_low_s":40,"hsv_low_v":40,
"hsv_high_h":175,"hsv_high_s":255,"hsv_high_v":255,"origin_x":37.543,"origin_y":-16.511,
"fov_x":228.5568,"fov_y":228.5568,"overlap":25.0,"clearance":750.0,"invert_x":True,"invert_y":False,
"z1":None,"z2":None,"z3":None,"plane":None,"plane_signature":None,"decimals":4}
for k,v in DEFAULTS.items(): ss.setdefault(k,v)
PAGES=["Dashboard","Image Upload & Perspective Correction","Tissue Boundary Detection","Coordinate Calibration","ROI Generation","XY PositionList","Z Reference Points","Plane Calculation","Z Coordinate Calculator","XYZ PositionList","3D Visualization"]
page=st.sidebar.radio("Workflow navigation",PAGES)
st.sidebar.divider()
st.sidebar.subheader("Acquisition settings")
ss.fov_x=st.sidebar.number_input("FOV X (µm)",min_value=1.0,value=float(ss.fov_x),step=0.1,format="%.4f")
ss.fov_y=st.sidebar.number_input("FOV Y (µm)",min_value=1.0,value=float(ss.fov_y),step=0.1,format="%.4f")
ss.overlap=st.sidebar.number_input("XY overlap (%)",min_value=0.0,max_value=99.0,value=float(ss.overlap),step=1.0)
ss.clearance=st.sidebar.number_input("Boundary clearance (µm)",min_value=0.0,value=float(ss.clearance),step=50.0)
st.sidebar.caption(f"Tile step: {ss.fov_x*(1-ss.overlap/100):.4f} × {ss.fov_y*(1-ss.overlap/100):.4f} µm")
st.sidebar.divider()
st.sidebar.subheader("Micro-Manager devices")
xy_device=st.sidebar.text_input("XY stage device",value=ss.get("xy_device","XYStage"),key="xy_device")
z_device=st.sidebar.text_input("Z stage device (exact name)",value=ss.get("z_device","ZStage"),key="z_device")
label_prefix=st.sidebar.text_input("Position label prefix",value=ss.get("label_prefix","ROI"),key="label_prefix")
template_upload=st.sidebar.file_uploader("Optional lab .pos template",type=["pos","txt","json"],key="template_upload")
if st.sidebar.button("Reset current tissue sample",type="secondary"):
    keep={k:v for k,v in ss.items() if k in ["fov_x","fov_y","overlap","clearance","invert_x","invert_y","origin_x","origin_y","hsv_low_h","hsv_low_s","hsv_low_v","hsv_high_h","hsv_high_s","hsv_high_v","xy_device","z_device","label_prefix"]}
    for k in list(ss.keys()): del ss[k]
    for k,v in DEFAULTS.items(): ss[k]=v
    ss.update(keep); st.rerun()

# Load and compute dependencies from persisted inputs on each rerun. This prevents stale downstream results.
def need(condition, message):
    if not condition: st.warning("Blocked by missing prerequisite: "+message); st.stop()
def get_image():
    image_bytes = ss.get("overview_image_bytes")
    if image_bytes:
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None
img=get_image()
# Explicitly invalidate selections when the uploaded image, perspective geometry or HSV detection changes.
image_hash=hashlib.sha256(ss.overview_image_bytes).hexdigest() if ss.get("overview_image_bytes") else None
if ss.get("last_image_hash") != image_hash:
    ss.last_image_hash=image_hash; ss.perspective_points=[]; ss.calibration_points=[]
    for i in (1,2,3): ss[f"z{i}"]=None
    ss.plane=None; ss.plane_signature=None
perspective_snapshot=tuple(tuple(p) for p in ss.perspective_points)
if ss.get("last_perspective_snapshot") is not None and ss.last_perspective_snapshot != perspective_snapshot:
    ss.calibration_points=[]; ss.plane=None; ss.plane_signature=None
ss.last_perspective_snapshot=perspective_snapshot
hsv_snapshot=tuple(ss.get(k) for k in ["hsv_low_h","hsv_low_s","hsv_low_v","hsv_high_h","hsv_high_s","hsv_high_v"])
if ss.get("last_hsv_snapshot") is not None and ss.last_hsv_snapshot != hsv_snapshot:
    ss.calibration_points=[]; ss.plane=None; ss.plane_signature=None
ss.last_hsv_snapshot=hsv_snapshot
corrected=None; contour=None; mask=None; boundary_xy=None; tile_df=pd.DataFrame(); stage_df=pd.DataFrame(); ref=None; plane=None
if img is not None and len(ss.perspective_points)==4:
    try: corrected,pm,_,_=mc.perspective_correct(img,ss.perspective_points)
    except Exception: corrected=None
if corrected is not None:
    mask,contour=mc.detect_purple_tissue(corrected,[ss.hsv_low_h,ss.hsv_low_s,ss.hsv_low_v],[ss.hsv_high_h,ss.hsv_high_s,ss.hsv_high_v])
if contour is not None and len(ss.calibration_points)==3:
    try:
        op,inv,cm,det=mc.calibrate_pixel_to_sample(ss.calibration_points,ss.origin_x,ss.origin_y)
        boundary_xy=np.asarray(mc.contour_to_new_coordinates(contour,op,inv,ss.origin_x,ss.origin_y),dtype=float)
    except Exception: boundary_xy=None
if boundary_xy is not None:
    try:
        tile_df,*tile_meta=mc.generate_roi_tiles(boundary_xy,ss.fov_x,ss.fov_y,ss.overlap,0.0,ss.clearance)
        if len(tile_df):
            xs=-1.0 if ss.invert_x else 1.0; ys=-1.0 if ss.invert_y else 1.0
            stage_df=tile_df.copy()
            stage_df["X_stage_um"]=stage_df["X_sample_mm"]*1000.0*xs
            stage_df["Y_stage_um"]=stage_df["Y_sample_mm"]*1000.0*ys
    except Exception as e: st.error(f"ROI generation error: {e}")

# Deterministic dependency signatures: changing inputs clears plane and calculated outputs.
def signature(keys):
    payload={k:ss.get(k) for k in keys}
    if ss.get("overview_image_bytes"): payload["image"]=hashlib.sha256(ss.overview_image_bytes).hexdigest()
    payload["perspective"]=ss.perspective_points; payload["calibration"]=ss.calibration_points
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
plane_sig=signature(["origin_x","origin_y","invert_x","invert_y","hsv_low_h","hsv_low_s","hsv_low_v","hsv_high_h","hsv_high_s","hsv_high_v","z1","z2","z3"])
roi_sig=signature(["fov_x","fov_y","overlap","clearance"])
if ss.get("last_plane_inputs")!=plane_sig:
    ss.last_plane_inputs=plane_sig; ss.plane=None; ss.plane_signature=None; ss.pop("xyz_bytes",None); ss.pop("xyz_df",None); ss.pop("xyz_checks",None)
if ss.get("last_roi_inputs")!=roi_sig:
    ss.last_roi_inputs=roi_sig; ss.pop("xyz_bytes",None); ss.pop("xyz_df",None); ss.pop("xyz_checks",None)
sig=plane_sig

def make_xy_positionlist(stage_positions_df, template_bytes=None):
    """Build XY list, preserving an uploaded lab template's per-position metadata when supplied."""
    if template_bytes is None:
        return mc.generate_pos(stage_positions_df.to_dict("records"), xy_device, label_prefix)
    template = pos.parse_pos(template_bytes.decode("utf-8"))
    original_positions = template["map"]["StagePositions"]["array"]
    if len(original_positions) != len(stage_positions_df):
        raise pos.PosFormatError(f"Uploaded template contains {len(original_positions)} positions but ROI generation produced {len(stage_positions_df)}. Counts must match to preserve the template's labels and metadata safely.")
    import copy
    merged = copy.deepcopy(template)
    for i, (_, row) in enumerate(stage_positions_df.iterrows()):
        sp = merged["map"]["StagePositions"]["array"][i]
        entries = sp["DevicePositions"]["array"]
        xy = next((e for e in entries if e.get("Device",{}).get("scalar")==xy_device and len(e.get("Position_um",{}).get("array",[]))==2), None)
        if xy is None:
            raise pos.PosFormatError(f"Position {i+1} has no two-axis device named '{xy_device}'. Choose the exact XY device name or use a compatible template.")
        xy["Position_um"]["array"]=[float(row["X_stage_um"]),float(row["Y_stage_um"])]
    return pos.dumps(merged).encode("utf-8")

# ------------------------------- pages
if page=="Dashboard":
    st.header("Workflow dashboard")
    metrics=st.columns(4)
    metrics[0].metric("Overview image","Loaded" if img is not None else "Missing")
    metrics[1].metric("Tissue boundary","Detected" if contour is not None else "Pending")
    metrics[2].metric("ROI positions",len(stage_df))
    metrics[3].metric("Plane","Calculated" if ss.plane is not None and ss.plane_signature==sig else "Pending")
    more=st.columns(4)
    more[0].metric("Z references entered",f"{sum(ss.get(f'z{i}') is not None for i in (1,2,3))}/3")
    more[1].metric("Valid Z coordinates",len(ss.get("xyz_df",[])) if ss.get("xyz_valid") else 0)
    more[2].metric("POS validation","PASS" if ss.get("xyz_valid") else "Not validated")
    more[3].metric("Output positions",len(ss.get("xyz_df",[])) if ss.get("xyz_valid") else 0)
    if ss.plane is not None and ss.plane_signature==sig:
        p=ss.plane; st.latex(rf"{p.A:.5g}x+{p.B:.5g}y+{p.C:.5g}z+{p.D:.5g}=0")
    steps=[("Image upload",img is not None), ("Perspective correction",corrected is not None), ("Purple boundary",contour is not None), ("Calibration",boundary_xy is not None), ("ROI + XY positions",len(stage_df)>0), ("Z reference measurements",all(ss.get(f"z{i}") is not None for i in (1,2,3))), ("Plane calculation",ss.plane is not None and ss.plane_signature==sig), ("Validated XYZ output",bool(ss.get("xyz_bytes")))]
    for name,done in steps:
        st.write(("🟢" if done else "⚪")+"  "+name+(" — Completed" if done else " — Not completed"))
    st.info("This application generates position-list files only. It does not control or move the microscope.")
    st.markdown("**Coordinate integrity:** local-origin calibration → existing origin shift → transformed sample coordinates (mm) → existing axis-direction sign → stage coordinates (µm). The original sequence is retained.")
elif page=="Image Upload & Perspective Correction":
    st.header("1. Image upload & perspective correction")
    up=st.file_uploader("Upload microscope overview image",type=["png","jpg","jpeg","tif","tiff"],key="overview_upload_widget")
    if up is not None:
        new_bytes = up.getvalue()
        if ss.get("overview_image_bytes") != new_bytes:
            ss.overview_image_bytes = new_bytes
            st.rerun()
    if img is not None:
        st.success(f"Image loaded: {img.shape[1]} × {img.shape[0]} pixels")
        st.image(mc.cv_rgb(img),caption="Original overview image",use_container_width=True)
        st.write("Select four corners in order: P1 top-left → P2 top-right → P3 bottom-right → P4 bottom-left.")
        pts=mc.click_selector(img,"perspective_points",4,"Click the four corners in the specified order.")
        if pts: st.image(mc.cv_rgb(mc.overlay_points(img,pts)),caption="Selected perspective points",use_container_width=True)
        if len(ss.perspective_points)==4 and corrected is not None:
            st.success(f"Perspective correction ready: {corrected.shape[1]} × {corrected.shape[0]} pixels")
            st.image(mc.cv_rgb(corrected),caption="Perspective-corrected image",use_container_width=True)
            st.download_button("Download corrected image",mc.jpg_bytes(corrected),"corrected_overview.jpg","image/jpeg")
        if st.button("Clear perspective points"): ss.perspective_points=[]; st.rerun()
    else: st.info("Upload an overview image to begin.")
elif page=="Tissue Boundary Detection":
    need(corrected is not None,"Upload an image and select all four perspective corners.")
    st.header("2. Purple tissue boundary detection")
    a,b=st.columns(2)
    with a:
        ss.hsv_low_h=st.number_input("H lower",0,179,int(ss.hsv_low_h)); ss.hsv_low_s=st.number_input("S lower",0,255,int(ss.hsv_low_s)); ss.hsv_low_v=st.number_input("V lower",0,255,int(ss.hsv_low_v))
    with b:
        ss.hsv_high_h=st.number_input("H upper",0,179,int(ss.hsv_high_h)); ss.hsv_high_s=st.number_input("S upper",0,255,int(ss.hsv_high_s)); ss.hsv_high_v=st.number_input("V upper",0,255,int(ss.hsv_high_v))
    mask,contour=mc.detect_purple_tissue(corrected,[ss.hsv_low_h,ss.hsv_low_s,ss.hsv_low_v],[ss.hsv_high_h,ss.hsv_high_s,ss.hsv_high_v])
    if contour is None: st.error("No purple tissue detected. Adjust HSV limits.")
    else:
        out=corrected.copy(); cv2.drawContours(out,[contour],-1,(0,0,255),2)
        st.success(f"Largest external contour detected · {len(contour)} boundary pixels · area {cv2.contourArea(contour):.1f} px²")
        c1,c2=st.columns(2); c1.image(mask,caption="HSV mask",use_container_width=True); c2.image(mc.cv_rgb(out),caption="Detected tissue boundary",use_container_width=True)
elif page=="Coordinate Calibration":
    need(corrected is not None and contour is not None,"Complete perspective correction and detect a purple tissue boundary.")
    st.header("3. Three-point pixel-to-sample calibration")
    st.markdown("P1 = local origin (0,0) · P2 = (0,30 mm) · P3 = (40,0 mm).")
    pts=mc.click_selector(corrected,"calibration_points",3,"Click P1 local origin, P2 (0,30 mm), then P3 (40,0 mm).")
    if pts: st.image(mc.cv_rgb(mc.overlay_points(corrected,pts,["P1 local origin","P2 (0,30)","P3 (40,0)"])),use_container_width=True)
    ss.origin_x=st.number_input("Microscope setup origin X (mm)",value=float(ss.origin_x),format="%.4f")
    ss.origin_y=st.number_input("Microscope setup origin Y (mm)",value=float(ss.origin_y),format="%.4f")
    if len(pts)==3:
        try:
            op,inv,cm,det=mc.calibrate_pixel_to_sample(pts,ss.origin_x,ss.origin_y)
            boundary_xy=np.asarray(mc.contour_to_new_coordinates(contour,op,inv,ss.origin_x,ss.origin_y))
            st.success("Calibration valid")
            st.code(np.array2string(cm,precision=8)); st.write(f"Determinant: {det:.8e}")
            st.write("**Transformation retained:** first calculate coordinates relative to the original local origin, then apply the microscope setup origin shift exactly once.")
            bd=pd.DataFrame(boundary_xy,columns=["X_new_mm","Y_new_mm"]); st.dataframe(bd.head(20),use_container_width=True)
            st.download_button("Download transformed tissue boundary CSV",bd.to_csv(index=False),"tissue_boundary_new_coordinates.csv","text/csv")
        except Exception as e: st.error(str(e))
    if st.button("Reset calibration points"): ss.calibration_points=[]; st.rerun()
elif page=="ROI Generation":
    need(boundary_xy is not None,"Complete valid three-point calibration.")
    st.header("4. ROI tile generation")
    st.caption("Tiles are generated in the transformed sample coordinate system (X_new_mm/Y_new_mm). Boundary clearance expands the ROI-selection polygon only; it does not alter the original tissue boundary.")
    if len(tile_df):
        fovx=ss.fov_x/1000; fovy=ss.fov_y/1000
        preview=mc.make_tile_preview(boundary_xy,list(zip(tile_df.X_sample_mm,tile_df.Y_sample_mm)),fovx,fovy,ss.clearance)
        st.image(mc.cv_rgb(preview),caption="Red: original tissue · Blue: clearance-expanded scan boundary · Green: tile footprints",use_container_width=True)
        st.metric("Generated ROI tile centres",len(tile_df)); st.dataframe(tile_df,use_container_width=True,height=360)
        st.download_button("Download ROI tile centres CSV",tile_df.to_csv(index=False),"roi_tile_centers.csv","text/csv")
    else: st.warning("No ROI tiles generated with current settings.")
elif page=="XY PositionList":
    need(len(stage_df)>0,"Generate ROI tile centres first.")
    st.header("5. Micro-Manager XY stage coordinates")
    st.markdown("The original sample transformation is already complete. Only the existing axis-direction signs and mm-to-µm conversion are applied here.")
    st.checkbox("Invert X axis (lab default: enabled)",key="invert_x")
    st.checkbox("Invert Y axis (lab default: disabled)",key="invert_y")
    xs=-1.0 if ss.invert_x else 1.0; ys=-1.0 if ss.invert_y else 1.0
    stage_df=tile_df.copy(); stage_df["X_stage_um"]=stage_df.X_sample_mm*1000*xs; stage_df["Y_stage_um"]=stage_df.Y_sample_mm*1000*ys
    st.code(f"X_stage_um = X_new_mm × 1000 × ({xs:+.0f})\nY_stage_um = Y_new_mm × 1000 × ({ys:+.0f})")
    st.dataframe(stage_df,use_container_width=True,height=380)
    st.image(mc.cv_rgb(mc.draw_stage_preview(stage_df.rename(columns={"X_stage_um":"X_stage_um","Y_stage_um":"Y_stage_um"}))),caption="XY stage scan-order preview",use_container_width=True)
    st.download_button("Download XY stage coordinates CSV",stage_df.to_csv(index=False),"ROI_PositionList_XY.csv","text/csv")
    if st.button("Generate XY .pos preview"):
        template=None
        if template_upload:
            try: template=json.loads(template_upload.getvalue().decode("utf-8"))
            except Exception as e: st.error(f"Template could not be read: {e}")
        try:
            b=mc.generate_pos(stage_df.to_dict("records"),xy_device,label_prefix,template=template)
            ss.xy_pos_bytes=b; st.success("XY PositionList generated.")
        except Exception as e: st.error(str(e))
    if ss.get("xy_pos_bytes"): st.download_button("Download XY PositionList",ss.xy_pos_bytes,"ROI_PositionList_XY.pos","application/json")
elif page=="Z Reference Points":
    need(boundary_xy is not None and len(stage_df)>0,"Complete calibration, ROI generation and XY stage conversion.")
    st.header("6. Automatic Z reference points")
    st.write("The three points are selected from the **original detected tissue contour**, not the clearance-expanded scan polygon. Focus the microscope at each displayed XY stage location and enter the measured Z-stage coordinate in µm.")
    pix,samp=mc.choose_three_boundary_points(contour,boundary_xy)
    xs=-1.0 if ss.invert_x else 1.0; ys=-1.0 if ss.invert_y else 1.0
    ref=pd.DataFrame({"Point":["Z1","Z2","Z3"],"Pixel_X":pix[:,0],"Pixel_Y":pix[:,1],"X_new_mm":samp[:,0],"Y_new_mm":samp[:,1],"X_stage_um":samp[:,0]*1000*xs,"Y_stage_um":samp[:,1]*1000*ys})
    marked=corrected.copy()
    colors=[(0,0,255),(0,180,0),(255,0,0)]
    for i,(x,y) in enumerate(pix):
        cv2.circle(marked,(int(round(x)),int(round(y))),12,colors[i],3); cv2.putText(marked,f"Z{i+1}",(int(x)+14,int(y)-14),cv2.FONT_HERSHEY_SIMPLEX,0.8,colors[i],2)
    st.image(mc.cv_rgb(marked),caption="Automatically selected Z1, Z2 and Z3 on the original tissue boundary",use_container_width=True)
    st.dataframe(ref,use_container_width=True)
    zcols=st.columns(3)
    for i,col in enumerate(zcols,1):
        raw=col.text_input(f"Measured Z{i} (µm)",value="" if ss[f"z{i}"] is None else str(ss[f"z{i}"]),key=f"z_input_{i}",placeholder="Enter measured Z in µm")
        try: ss[f"z{i}"]=float(raw) if raw.strip() else None
        except ValueError: ss[f"z{i}"]=None; col.error("Enter a numeric Z value.")
    st.caption("Enter the actual microscope Z-stage reading for each corresponding X/Y location. Do not enter image pixel coordinates or sample mm values.")
    st.session_state.reference_table=ref
    if st.button("Clear Z measurements"):
        for i in (1,2,3): ss[f"z{i}"]=None; ss.pop(f"z_input_{i}",None)
        ss.plane=None; st.rerun()
elif page=="Plane Calculation":
    need(boundary_xy is not None and len(stage_df)>0,"Complete image calibration and XY generation.")
    pix,samp=mc.choose_three_boundary_points(contour,boundary_xy); xs=-1.0 if ss.invert_x else 1.0; ys=-1.0 if ss.invert_y else 1.0
    need(all(ss.get(f"z{i}") is not None for i in (1,2,3)),"Enter all three measured Z coordinates on the Z Reference Points page.")
    refxy=samp*1000*np.array([xs,ys]); points=[(refxy[i,0],refxy[i,1],float(ss[f"z{i+1}"])) for i in range(3)]
    st.header("7. Original 3D plane calculation")
    st.write("Plane inputs are the final microscope stage coordinates in µm. The original local-origin calibration and microscope-origin transformation have already been applied exactly once.")
    st.dataframe(pd.DataFrame(points,columns=["X_stage_um","Y_stage_um","Measured Z_um"],index=["Z1","Z2","Z3"]))
    if st.button("Calculate Tissue Plane",type="primary"):
        try:
            p=calc.compute_plane(points)
            if not p.z_usable: raise calc.PlaneError("Plane is nearly vertical; Z cannot be reliably calculated from X/Y.")
            ss.plane=p; ss.plane_signature=sig; st.success("Plane calculated and verified using the original compute_plane() implementation.")
        except Exception as e: ss.plane=None; st.error(str(e))
    if ss.plane is not None and ss.plane_signature==sig:
        p=ss.plane
        st.latex(rf"{p.A:.8g}x+{p.B:.8g}y+{p.C:.8g}z+{p.D:.8g}=0")
        st.write(f"Solve for Z: `Z = -({p.A:.8g}X + {p.B:.8g}Y + {p.D:.8g}) / {p.C:.8g}`")
        st.markdown("**Step-by-step derivation (original calculation implementation)**")
        st.write("u = P2 − P1",p.u); st.write("v = P3 − P1",p.v); st.write("n = u × v",p.n)
        st.write(f"A = {p.A:.10g}, B = {p.B:.10g}, C = {p.C:.10g}")
        st.write(f"D = −(A·X1 + B·Y1 + C·Z1) = {p.D:.10g}")
        st.write("Reference-point residuals",p.residuals()); st.write(f"Residual tolerance: {p.tolerance():.4e}")
        st.write(f"sin(angle) = {p.sin_angle:.4e} · |C|/|n| = {p.c_ratio:.4e}")
elif page=="Z Coordinate Calculator":
    need(ss.plane is not None and ss.plane_signature==sig,"Calculate a valid tissue plane first.")
    st.header("8. Z coordinate calculator")
    p=ss.plane
    x=st.number_input("X stage (µm)",value=0.0); y=st.number_input("Y stage (µm)",value=0.0)
    st.metric("Predicted Z (µm)",f"{float(p.z_at(x,y)):.4f}")
    up=st.file_uploader("Optional CSV with X_stage_um and Y_stage_um",type=["csv"],key="z_csv")
    if up:
        try:
            df=pd.read_csv(up); need_cols={"X_stage_um","Y_stage_um"}
            if not need_cols.issubset(df.columns): st.error("CSV must contain X_stage_um and Y_stage_um.")
            else:
                df["Z_stage_um"]=p.z_at(df.X_stage_um.to_numpy(),df.Y_stage_um.to_numpy())
                edited=st.data_editor(df,use_container_width=True,num_rows="dynamic")
                if {"X_stage_um","Y_stage_um"}.issubset(edited.columns):
                    edited["Z_stage_um"]=p.z_at(edited.X_stage_um.to_numpy(),edited.Y_stage_um.to_numpy())
                    st.download_button("Download calculated Z CSV",edited.to_csv(index=False),"calculated_z.csv","text/csv")
        except Exception as e: st.error(str(e))
elif page=="XYZ PositionList":
    need(len(stage_df)>0,"Generate valid XY ROI positions first."); need(ss.plane is not None and ss.plane_signature==sig,"Calculate a valid plane first.")
    st.header("9. Final XYZ PositionList")
    p=ss.plane
    xyz=stage_df.copy(); xyz["Z_stage_um"]=p.z_at(xyz.X_stage_um.to_numpy(),xyz.Y_stage_um.to_numpy())
    xyz=xyz.rename(columns={"row":"Row","col":"Column","X_sample_mm":"X_sample_mm","Y_sample_mm":"Y_sample_mm"})
    st.dataframe(xyz,use_container_width=True,height=360)
    st.metric("ROI positions",len(xyz)); st.metric("Calculated Z range",f"{xyz.Z_stage_um.min():.4f} to {xyz.Z_stage_um.max():.4f} µm")
    # Extrapolation diagnostic against reference-point bounding box, matching original calculator behaviour.
    pix,samp=mc.choose_three_boundary_points(contour,boundary_xy); xs=-1.0 if ss.invert_x else 1.0; ys=-1.0 if ss.invert_y else 1.0; rxy=samp*1000*np.array([xs,ys])
    outside=(xyz.X_stage_um<rxy[:,0].min())|(xyz.X_stage_um>rxy[:,0].max())|(xyz.Y_stage_um<rxy[:,1].min())|(xyz.Y_stage_um>rxy[:,1].max())
    if outside.any(): st.warning(f"{int(outside.sum())} ROI positions lie outside the reference-point bounding box. Their Z values are extrapolations. A three-point plane is only a local approximation of the tissue surface.")
    else: st.success("All ROI positions lie within the reference-point bounding box. This does not prove the tissue surface is planar.")
    confirm_add_z=st.checkbox("Confirm adding the Z-stage device if it is missing from the template",value=False,key="confirm_add_z")
    if st.button("Generate and validate XYZ .pos",type="primary"):
        try:
            xy_bytes=make_xy_positionlist(stage_df, template_upload.getvalue() if template_upload else None)
            base=pos.parse_pos(xy_bytes.decode("utf-8"))
            updated,updated_count,added_count=pos.apply_z(base,xyz.Z_stage_um.tolist(),z_device,allow_add=confirm_add_z)
            out=pos.dumps(updated); checks=pos.validate_roundtrip(base,out,xyz.Z_stage_um.tolist(),z_device)
            ss.xyz_bytes=out; ss.xyz_checks=checks; ss.xyz_df=xyz; ss.xyz_valid=all(checks.values())
            st.success(f"POS processor updated {updated_count} Z entries and added {added_count} entries.")
        except Exception as e: st.error(f"XYZ generation/validation failed: {e}")
    if ss.get("xyz_checks") is not None:
        st.subheader("Round-trip validation")
        for k,v in ss.xyz_checks.items(): st.write(("✅ PASS" if v else "❌ FAIL")+f" — {k}")
        if ss.get("xyz_valid"):
            st.download_button("Download validated ROI_PositionList_XYZ.pos",ss.xyz_bytes,"ROI_PositionList_XYZ.pos","application/json")
            st.download_button("Download final XYZ CSV",ss.xyz_df.to_csv(index=False),"ROI_PositionList_XYZ.csv","text/csv")
            st.warning("Before real acquisition, inspect the generated list in Micro-Manager and verify labels, XY, Z, device names and safe stage limits. This application does not move the microscope.")
        else: st.error("Critical validation failed. Downloads are disabled.")
    with st.expander("Advanced: process an existing POS file independently"):
        st.write("This retains the original POS processor workflow. The uploaded list's own XY stage coordinates are used with the currently calculated plane; no image-derived XY replacement is performed in this mode.")
        pos_file=st.file_uploader("Upload existing Micro-Manager 2.x .pos",type=["pos","txt","json"],key="standalone_pos_upload")
        if pos_file:
            try:
                source=pos.parse_pos(pos_file.getvalue().decode("utf-8"))
                rows=pos.extract_table(source)
                st.dataframe(pd.DataFrame(rows),use_container_width=True)
                standalone_z=ss.plane.z_at(np.array([r["X"] for r in rows]),np.array([r["Y"] for r in rows]))
                st.write(f"Calculated Z for {len(rows)} existing positions.")
                confirm_standalone_add=st.checkbox("Confirm adding missing Z device if it is absent",key="standalone_confirm_add")
                if st.button("Update uploaded POS with calculated Z",key="standalone_apply_z"):
                    updated,n_upd,n_add=pos.apply_z(source,standalone_z.tolist(),z_device,allow_add=confirm_standalone_add)
                    out=pos.dumps(updated); checks=pos.validate_roundtrip(source,out,standalone_z.tolist(),z_device)
                    st.session_state.standalone_pos_result=(out,checks)
                if ss.get("standalone_pos_result"):
                    out,checks=ss.standalone_pos_result
                    for k,v in checks.items(): st.write(("✅ PASS" if v else "❌ FAIL")+" — "+k)
                    if all(checks.values()): st.download_button("Download processed existing POS",out,"existing_positions_with_calculated_Z.pos","application/json")
                    else: st.error("Validation failed; download disabled.")
            except Exception as e: st.error(f"Existing POS processing failed: {e}")

elif page=="3D Visualization":
    need(ss.plane is not None and ss.plane_signature==sig,"Calculate a valid plane first.")
    st.header("10. 3D tissue plane visualization")
    true=st.checkbox("Use true-scale axes",value=False)
    st.plotly_chart(viz.make_figure(ss.plane,unit="µm",true_scale=true),use_container_width=True)
    st.caption("The visualization shows the measured reference points, fitted plane and normal vector. The fitted plane is a local approximation, not proof of a globally planar tissue surface.")
