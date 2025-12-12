"""Streamlit app for qualitative evaluation of rock joint segmentation results.

This app allows raters to view epoch progression images one-by-one and provide
qualitative ratings across multiple criteria. Ratings are saved to a CSV file
for later analysis.

Usage:
    streamlit run scripts/results_analysis/qualitative_evaluation_app.py

"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
STAGES = [
    "epoch_5",
    "epoch_10",
    "epoch_13",
    "epoch_17",
    "final",
]

CRITERIA = [
    (
        "geological_recognisability",
        "Geological recognisability",
        "How realistic and geologically plausible do the predicted joints appear?",
    ),
    (
        "joint_persistence",
        "Joint persistence",
        "Are continuous joints properly connected without gaps or interruptions?",
    ),
    (
        "boundary_localisation",
        "Boundary localisation & thickness",
        "Are joint boundaries precise and is the thickness appropriate?",
    ),
    (
        "false_positives",
        "False positives / noise",
        "How much spurious segmentation or noise is present? (1=very noisy, 5=very clean)",
    ),
]

SCORES = [1, 2, 3, 4, 5]
SCORE_LABELS = "1 = Poor | 2 = Fair | 3 = Acceptable | 4 = Good | 5 = Excellent"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


# -----------------------------
# Helpers
# -----------------------------
def now():
    return datetime.now().isoformat(timespec="seconds")


def load_df(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def upsert(df, row):
    if df.empty or "image" not in df.columns or "stage" not in df.columns:
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    key = (df.image == row["image"]) & (df.stage == row["stage"])
    if not key.any():
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.loc[key, :] = pd.DataFrame([row]).values
    return df


# -----------------------------
# App
# -----------------------------
st.set_page_config(layout="wide")
st.title("Rock Joint Segmentation – Qualitative Evaluation")

with st.sidebar:
    default_image_folder = "experiments/results/plots/progression"
    data_dir = Path(
        st.text_input("Image folder", value=default_image_folder)
    ).expanduser()
    rater = st.text_input("Rater ID", value="rater_1")

    default_csv_path = "experiments/results/qualitative_ratings.csv"
    out_csv = Path(st.text_input("Output CSV", value=default_csv_path))
    st.markdown("---")

if not data_dir.exists():
    st.stop()

images = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

if not images:
    st.error(f"No images found in {data_dir}. Please check the folder path.")
    st.stop()

df = load_df(out_csv)

if "idx" not in st.session_state:
    st.session_state.idx = 0

# Ensure idx is within valid range
if st.session_state.idx >= len(images):
    st.session_state.idx = 0
elif st.session_state.idx < 0:
    st.session_state.idx = len(images) - 1

# Show progress overview in sidebar
with st.sidebar:
    st.markdown("### Progress Overview")

    if not df.empty and "image" in df.columns and "rater" in df.columns:
        # Get rated images for current rater
        rater_df = df[df.rater == rater]
        rated_images = set(rater_df.image.unique()) if not rater_df.empty else set()
    else:
        rated_images = set()

    total_images = len(images)
    rated_count = len([img for img in images if img.name in rated_images])
    progress = rated_count / total_images if total_images > 0 else 0

    st.progress(
        progress,
        text=f"{rated_count}/{total_images} images rated ({progress * 100:.0f}%)",
    )

    st.markdown("---")
    st.markdown("**Images:**")
    for i, img in enumerate(images):
        status = "✓" if img.name in rated_images else "○"
        label = f"{status} {i + 1}. {img.name[:30]}..."
        if st.button(label, key=f"nav_{i}", use_container_width=True):
            st.session_state.idx = min(i, len(images) - 1)
            st.rerun()

img_path = images[st.session_state.idx]

# Initialize keyboard navigation state
if "nav_action" not in st.session_state:
    st.session_state.nav_action = None

# Keyboard navigation and quick rating handler
keyboard_script = """
<script>
const doc = window.parent.document;
let quickRatingMode = false;
let quickRatings = [];
const criteriaNames = [
    '1️⃣ Geological recognisability',
    '2️⃣ Joint persistence',
    '3️⃣ Boundary localisation',
    '4️⃣ False positives'
];

// Create status indicator
let statusDiv = doc.getElementById('quick-rating-status');
if (!statusDiv) {
    statusDiv = doc.createElement('div');
    statusDiv.id = 'quick-rating-status';
    statusDiv.style.cssText = 'position: fixed; top: 80px; right: 20px; background: rgba(0, 150, 255, 0.95); color: white; padding: 15px 20px; border-radius: 8px; font-size: 16px; font-weight: bold; z-index: 9999; display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-family: monospace;';
    doc.body.appendChild(statusDiv);
}

function updateStatus(message, show = true) {
    statusDiv.innerHTML = message;
    statusDiv.style.display = show ? 'block' : 'none';
    if (show && !quickRatingMode) {
        setTimeout(() => { statusDiv.style.display = 'none'; }, 2000);
    }
}

doc.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;  // Don't trigger when typing in form fields
    }

    // Quick rating mode: r followed by 4 digits
    if (e.key.toLowerCase() === 'r' && !quickRatingMode) {
        quickRatingMode = true;
        quickRatings = [];
        console.log('Quick rating mode activated. Enter 4 numbers (1-5)...');
        updateStatus('🎯 Quick Rating Mode<br><br>' + criteriaNames[0]);
        return;
    }

    if (quickRatingMode) {
        const num = parseInt(e.key);
        if (num >= 1 && num <= 5) {
            quickRatings.push(num);
            console.log(`Captured rating ${quickRatings.length}: ${num}`);

            // Show next criterion or completion
            if (quickRatings.length < 4) {
                const prevCrit = criteriaNames[quickRatings.length - 1].replace(/[0-9]️⃣ /, '');
                const nextCrit = criteriaNames[quickRatings.length];
                updateStatus('🎯 Quick Rating Mode<br><br>✓ ' + prevCrit + ': ' + num + '<br><br>' + nextCrit);
            }

            if (quickRatings.length === 4) {
                // Find all radio button groups (4 criteria) - search more thoroughly
                let radioGroups = doc.querySelectorAll('[role="radiogroup"]');

                // If not found, search in all iframes (Streamlit sometimes uses iframes)
                if (radioGroups.length < 4) {
                    const iframes = doc.querySelectorAll('iframe');
                    for (let iframe of iframes) {
                        try {
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            const iframeGroups = iframeDoc.querySelectorAll('[role="radiogroup"]');
                            if (iframeGroups.length >= 4) {
                                radioGroups = iframeGroups;
                                break;
                            }
                        } catch (e) {
                            // Skip iframes we can't access due to CORS
                        }
                    }
                }

                // Also try finding by data-testid or class patterns
                if (radioGroups.length < 4) {
                    radioGroups = doc.querySelectorAll('div[data-testid*="stRadio"]');
                }

                console.log('Found radio groups:', radioGroups.length);

                if (radioGroups.length >= 4) {
                    let successCount = 0;
                    for (let i = 0; i < 4; i++) {
                        // Find all radio inputs in this group
                        const radios = radioGroups[i].querySelectorAll('input[type="radio"]');
                        console.log(`Group ${i}: found ${radios.length} radio buttons`);

                        if (radios.length >= quickRatings[i]) {
                            // Click the appropriate radio button (quickRatings[i] is 1-5, array is 0-4)
                            const targetRadio = radios[quickRatings[i] - 1];
                            targetRadio.click();
                            // Also trigger change event to ensure Streamlit registers it
                            targetRadio.dispatchEvent(new Event('change', { bubbles: true }));
                            successCount++;
                            console.log(`Clicked radio ${i}: value ${quickRatings[i]}`);
                        }
                    }
                    console.log(`Quick ratings applied: ${successCount}/4 successful`, quickRatings);
                    updateStatus('✅ Ratings Applied!<br><br>' + quickRatings.join(' - '));
                } else {
                    console.warn('Could not find enough radio groups. Found:', radioGroups.length);
                    updateStatus('⚠️ Could not find radio buttons', true);
                }

                quickRatingMode = false;
                quickRatings = [];
            }
        } else {
            // Invalid input, cancel quick rating mode
            quickRatingMode = false;
            quickRatings = [];
            updateStatus('❌ Quick Rating Cancelled', true);
            console.log('Quick rating mode cancelled');
        }
        return;
    }

    // Ctrl+S to save
    if (e.ctrlKey && e.key.toLowerCase() === 's') {
        e.preventDefault();  // Prevent browser save dialog
        const saveBtn = Array.from(doc.querySelectorAll('button')).find(btn =>
            btn.innerText === 'Save' || (btn.innerText.includes('Save') && !btn.innerText.includes('Next'))
        );
        if (saveBtn) {
            saveBtn.click();
            console.log('Triggered Save (Ctrl+S)');
        }
        return;
    }

    // Ctrl+Q to save & next
    if (e.ctrlKey && e.key.toLowerCase() === 'q') {
        e.preventDefault();
        const saveNextBtn = Array.from(doc.querySelectorAll('button')).find(btn =>
            btn.innerText.includes('Save') && btn.innerText.includes('Next')
        );
        if (saveNextBtn) {
            saveNextBtn.click();
            console.log('Triggered Save & Next (Ctrl+Q)');
        }
        return;
    }

    // Arrow key navigation
    if (e.key === 'ArrowLeft') {
        const prevBtn = Array.from(doc.querySelectorAll('button')).find(btn => btn.innerText.includes('Prev'));
        if (prevBtn) prevBtn.click();
    } else if (e.key === 'ArrowRight') {
        const nextBtn = Array.from(doc.querySelectorAll('button')).find(btn => btn.innerText.includes('Next'));
        if (nextBtn) nextBtn.click();
    }
});
</script>
"""
components.html(keyboard_script, height=0)

# Navigation
colA, colB, colC = st.columns([1, 2, 1])
with colA:
    if st.button("◀ Prev (←)") and st.session_state.idx > 0:
        st.session_state.idx -= 1
        st.rerun()
with colB:
    st.caption("💡 **R**+4nums: Quick rate | **Ctrl+S**: Save | **Ctrl+Q**: Save&Next")
with colC:
    if st.button("Next (→) ▶") and st.session_state.idx < len(images) - 1:
        st.session_state.idx += 1
        st.rerun()

# Check if current image has been rated
stage = "final"
is_rated = False
if not df.empty and "image" in df.columns and "stage" in df.columns:
    is_rated = (
        (df.image == img_path.name) & (df.stage == stage) & (df.rater == rater)
    ).any()

# Display title with green checkmark if rated
status_icon = " :green[✓]" if is_rated else ""
st.markdown(
    f"### Image {st.session_state.idx + 1} / {len(images)}{status_icon} — `{img_path.name}`"
)

# Two-column layout: image on left, evaluation form on right
col_img, col_form = st.columns([1.2, 1])

with col_img:
    # Display image at full resolution
    img = Image.open(img_path)
    st.image(img, use_container_width=True)

with col_form:
    st.markdown("## Evaluation - Final Epoch")

    existing = {}
    if not df.empty and "image" in df.columns and "stage" in df.columns:
        matches = df[
            (df.image == img_path.name) & (df.stage == stage) & (df.rater == rater)
        ]
        if not matches.empty:
            existing = matches.iloc[0].to_dict()

    with st.form(f"{img_path.name}_{stage}"):
        st.caption(f"**Rating Scale:** {SCORE_LABELS}")
        st.markdown("")  # Spacing
        scores = {}
        for key, label, description in CRITERIA:
            st.markdown(f"**{label}**")
            st.caption(description)
            default = int(existing.get(key, 3)) if existing else 3
            scores[key] = st.radio(
                f"{key}_radio",
                SCORES,
                index=SCORES.index(default),
                horizontal=True,
                label_visibility="collapsed",
            )

        st.markdown("---")
        st.markdown("**Optional: Earlier epoch comparison**")

        # Parse existing better_epoch value if it exists
        existing_epochs = []
        if existing.get("better_epoch"):
            val = existing.get("better_epoch")
            if isinstance(val, list):
                existing_epochs = val
            elif isinstance(val, str):
                existing_epochs = [e.strip() for e in val.split(",") if e.strip()]

        # Detect strategy from current image filename
        img_name_lower = img_path.name.lower()

        if "finetune" in img_name_lower:
            # Finetune strategy: Epoch 5, 10, Stage 2 Start, Stage 2 5th
            epoch_options = [
                ("epoch_5", "Epoch 5"),
                ("epoch_10", "Epoch 10"),
                ("epoch_13", "Stage 2 Start"),
                ("epoch_17", "Stage 2 5th"),
            ]
            strategy_detected = "finetune"
        elif "simplemixed" in img_name_lower or "simple" in img_name_lower:
            # SimpleMixed strategy: Epoch 5, 10, 15, 20
            epoch_options = [
                ("epoch_5", "Epoch 5"),
                ("epoch_10", "Epoch 10"),
                ("epoch_15", "Epoch 15"),
                ("epoch_20", "Epoch 20"),
            ]
            strategy_detected = "simplemixed"
        else:
            # Default: use all available epochs except final
            epoch_options = [
                (s, s.replace("_", " ").title()) for s in STAGES if s != "final"
            ]
            strategy_detected = "unknown"

        # Extract keys and labels
        epoch_keys = [key for key, _ in epoch_options]
        epoch_labels = dict(epoch_options)

        # Filter existing epochs to only include valid options for this strategy
        existing_epochs = [e for e in existing_epochs if e in epoch_keys]

        selected_epochs = st.multiselect(
            "Best epoch(s) shown here that outperform final in general:",
            options=epoch_keys,
            format_func=lambda x: epoch_labels.get(x, x),
            default=existing_epochs,
            help="Leave empty if final epoch is best overall",
        )

        better_epoch = ", ".join(selected_epochs) if selected_epochs else ""

        notes = st.text_area(
            "Notes (optional)", value=existing.get("notes", ""), height=80
        )

        col_save, col_save_next = st.columns(2)
        with col_save:
            save_clicked = st.form_submit_button(
                "Save (Ctrl+S)", use_container_width=True
            )
        with col_save_next:
            save_next_clicked = st.form_submit_button(
                "Save & Next → (Ctrl+Q)", use_container_width=True, type="primary"
            )

        if save_clicked or save_next_clicked:
            row = {
                "timestamp": now(),
                "rater": rater,
                "image": img_path.name,
                "stage": stage,
                **scores,
                "better_epoch": better_epoch,
                "notes": notes,
            }
            df = upsert(df, row)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            st.success(f"Saved {stage}")

            # Navigate to next image if Save & Next was clicked
            if save_next_clicked and st.session_state.idx < len(images) - 1:
                st.session_state.idx += 1
                st.rerun()

    st.caption(f"Saved to {out_csv.resolve()}")
