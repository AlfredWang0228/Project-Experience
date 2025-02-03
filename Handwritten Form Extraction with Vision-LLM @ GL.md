# 1. Motivation & Background

> **Summary:**  
> We aim to replace or augment our legacy OCR solution (e.g., AWS Textract) with a Vision-Language Large Language Model (VLLM) to more reliably parse hand-written forms. Below, we outline the high-level business needs, challenges, and objectives that led us to undertake this project.

---

## Business Context

- **Form Volume & Criticality**  
  - We handle thousands of hand-written forms daily (e.g., insurance applications). Even a single digit error in key fields can lead to serious downstream consequences (claims misprocessing, legal risks, customer dissatisfaction).  
  - Reliance on manual checks is expensive and slow. Errors discovered late cause **rework**, **delays**, and **customer complaints**.

- **Limitations of Legacy OCR**  
  - **AWS Textract** is our current go-to OCR tool; it handles typed text reasonably well but struggles with messy or stylized handwriting.  
  - Frequent misreads necessitate **intensive manual verification** (employees re-check extracted text, field by field).  
  - Inconsistencies in extracted data can have substantial impacts on regulated or sensitive fields (e.g., policy numbers).

> **Key Pain Point:** A single misread digit in a policy number can trigger a significant cascade of errors and customer complaints.

---

## Technical Rationale

- **Why a Vision-LLM?**  
  1. **Contextual Understanding**  
     - Traditional OCR sees text purely character-by-character. A Vision-LLM can leverage **spatial** and **semantic** cues—particularly helpful for structured forms.
  2. **Unified Extract-Interpret Pipeline**  
     - Instead of performing OCR and then a separate parsing step, a VLLM can directly generate JSON.  
     - This reduces complexity (avoiding mismatch issues between OCR output and downstream parsers).
  3. **Robustness to Handwriting Variations**  
     - Handwriting can differ drastically by individual. Vision-LLMs, especially with contrastive and augmentation techniques, can learn more robust representations of text in images.

- **Objective**  
  - **Minimize Manual Corrections:** Dramatically reduce human intervention by ensuring high accuracy for critical fields.  
  - **Improve Field-Level Accuracy:** Lower error rates from as high as 20% to under 5% for challenging handwritten entries.

---

## Key Challenges

1. **High Variability in Handwriting**  
   - Letters may be connected or inconsistently formed.  
   - Ink smudges, scanning artifacts, or poor-quality images reduce clarity.

2. **Multiple Form Layouts**  
   - Each form variant has unique layouts and field placements.  
   - Enforcing a consistent JSON schema requires robust detection of the correct field positions.

3. **Limited Labeled Data**  
   - High-quality real scans with verified annotations are expensive to obtain in large quantities.  
   - Synthetic data generation and augmentation help fill these gaps.

4. **Resource Constraints**  
   - Full fine-tuning of large foundation models can be prohibitive.  
   - **LoRA** (Low-Rank Adaptation) helps reduce compute requirements by focusing updates on a small subset of parameters.

---

## Impact & Business Value

| **Factor**                        | **Legacy OCR + Manual Checks**                               | **VLLM-Based Extraction**                                    |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Accuracy**                      | Moderate; heavily reliant on staff for review & correction   | Significantly improved with structured JSON output (less manual intervention) |
| **Operational Costs**             | High, due to labor-intensive verification                    | Lower ongoing costs after initial fine-tuning                |
| **Scalability**                   | Doesn’t scale well (manual processes balloon with data volume) | Automated pipeline scales easily with volume                 |
| **Complex Layouts & Handwriting** | Often fails for messy handwriting; requires brittle rules-based fixes | End-to-end model learns from diverse samples; robust to distortions & style changes |

> **Bottom Line:** By improving accuracy and automating extraction, we reduce manual rework time, boost process efficiency, and enhance customer satisfaction.

---

## Project Goals

1. **Reduce Error Rates**  
   - Attain **<2%** field-level error rate for critical fields (e.g., policy number, name) in test scenarios.

2. **Lower Manual Verification**  
   - Decrease the time staff spend on validation and corrections, ideally cutting it by half or more.

3. **Elevate Customer Experience**  
   - Fewer mistakes lead to fewer complaints, improved trust, and better overall service.

4. **Scalable & Maintainable Pipeline**  
   - Incorporate form clustering (via ResNet-based similarity search) and synthetic data strategies to streamline the addition of new form types.

---

# 2. Data Source & Preprocessing

> **Summary:**  
> Our data comes from three primary channels—real scanned forms, synthetic documents, and augmented variations—to ensure robust coverage of handwriting and layout variations. This section details how each source is obtained and the key preprocessing steps (including form-type clustering) that prepare data for model training.

---

## Data Sources

### 2.1 Original Scanned Images
- **Real-World Forms**: These are actual, hand-written forms received from customers.  
- **Challenges**:  
  - Variability in ink, scanning quality, and handwriting style.  
  - Potentially incomplete or inconsistent labeling for certain fields.  

> **Key Value**: Ground-truth annotations from real forms are the **most critical** for ensuring final model accuracy on real-world use cases.

---

### 2.2 Synthetic Data
- **Motivation**: Large-scale labeled data for hand-written forms is costly and time-consuming to collect.  
- **Approach**:  
  - We use an LLM to generate “fake” but format-consistent data (e.g., random names, addresses, policy numbers).  
  - The content is then rendered into PDF or image formats, mimicking real forms.  
- **Advantages**:  
  - Allows scaling up the dataset quickly, covering many field variations.  
  - Ensures perfect labels, since we control every generated field.

> **Caution**: Synthetic data may not capture all the nuances (e.g., truly messy handwriting), so it’s used **in tandem** with real data, not as a complete substitute.

---

### 2.3 Augmented Original Data
- **Objective**: Boost model robustness to scanning artifacts and handwriting variations.  
- **Augmentation Techniques**:  
  1. **Geometric** (rotation, random cropping, mild perspective distortion)  
  2. **Photometric** (brightness, contrast, blur, noise)  
  3. **Handwriting Style Alterations** (if feasible, e.g., digital warping to simulate certain pen strokes)  
- **Usage**:  
  - We apply these augmentations to real scanned images, retaining the original ground-truth labels.  
  - By training on multiple distorted versions of the same form, the model learns invariances that reduce errors on poor-quality scans.

> **Note**: Augmentation must be carefully tuned so that it doesn’t produce illegible text, which could degrade model performance.

---

## Preprocessing Steps

### 2.4 Form Clustering & Classification (ResNet)
- **Step 1**: **Feature Extraction**  
  - We employ a pretrained ResNet to generate embeddings for each form image.  
- **Step 2**: **Similarity Search**  
  - Images are grouped or clustered based on feature similarity (e.g., k-means or nearest-neighbor search).  
  - Each cluster represents a *form type* (e.g., “Form A, Page 1”). This classification helps us route each form to the appropriate JSON schema.  
- **Step 3**: **Group Label Assignment**  
  - A subject matter expert or a semi-automated script confirms the cluster’s label (e.g., “Application Page 2”).

> **Benefit**: This automated grouping accelerates labeling and ensures consistent output formats for each distinct form layout.

---

### 2.5 Image Normalization
- **Cropping & Resizing**: Standardize resolution to match the Vision-LLM’s input requirements.  
- **Color/Grayscale Conversion**: Convert images to RGB or grayscale as needed for the model.  
- **Masking or Redaction** (if applicable): Some fields (e.g., sensitive IDs) may be masked or partially redacted to comply with privacy policies.

---

### 2.6 Label Preparation
- **JSON Schema Definition**: Each form type is associated with a predefined set of fields (e.g., “policy_number,” “customer_name,” etc.).  
- **Annotation Process**:  
  - For real data, human annotators validate or correct automatically extracted fields.  
  - For synthetic data, labels are generated automatically from the known “fake” content.  
  - For augmented data, original labels remain valid since augmentation preserves text content.

> **Important**: **Quality control**—a small portion of data is rechecked by humans to ensure labeling consistency and correctness.

---

### 2.7 Data Pairing for Contrastive Learning
- **Positive Pairs**:  
  - Original image vs. its augmented counterpart.  
  - Synthetic form vs. a mildly distorted version of the same synthetic form.  
- **Negative Pairs**:  
  - Forms of different layout types or drastically different content.  
- **Usage**:  
  - The model is encouraged to learn similar embeddings for visually similar documents (positive pairs) and distinct embeddings for dissimilar ones (negative pairs).  
  - Enhances robustness against minor distortions and reduces confusion across different form types.

---

## Summary of Preprocessing Pipeline

1. **Acquire & Label Data**: Combine real scanned forms, synthetic forms, and augmented variants.  
2. **Cluster with ResNet**: Group forms by type/layout to facilitate schema-based extraction.  
3. **Normalize & Clean**: Resize, crop, and ensure consistent color channels.  
4. **Set JSON Schema**: Map each form type to its corresponding fields and structure.  
5. **Pair Data for Contrastive**: Organize positive/negative pairs to support the hybrid CE + contrastive training.  

> **Outcome**: A comprehensive dataset that covers a wide range of handwriting styles and form variations, ensuring the Vision-LLM model trains on both real-world and synthetic scenarios for maximum accuracy.



# 3. Model Architecture & Technical Approach

> **Summary:**
>  We employ **LLaVA 1.6** as our base Vision-Language Model (VLLM), focusing fine-tuning on the **projection layer** and the **top few layers** of the vision encoder. The training objective combines **Cross-Entropy (CE)** for structured JSON output with a **contrastive loss** to make embeddings robust to image distortions. Details on the model’s components, parameter-efficient tuning, and key hyperparameters follow.

------

## 3.1 Model Choice & Overview

- Base Model: LLaVA 1.6
  - A multi-modal model built on a LLaMA-family language model plus a vision encoder (often ViT-based or CLIP-based).
  - A “projection” or “Q-Former” layer bridges the visual features to the language model’s token embeddings.

> **Why LLaVA 1.6?**
>
> - Strong open-source baseline for image-to-text tasks.
> - Supports **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning.
> - Backed by an active community and existing benchmarks.

------

## 3.2 Fine-Tuning Approach

1. **Parameter-Efficient Strategy (LoRA)**
   - Freeze most of the model.
   - Unfreeze only the **top few layers** of the vision encoder (to adapt to handwriting) and the **projection layer**.
   - Insert LoRA modules into these layers, reducing GPU memory usage compared to full fine-tuning.
2. **Single vs. Multi-Form**
   - **Single-Model Strategy**: Use a shared model for all form types, but prepend a form-specific token or prompt.
   - **Multi-Checkpoint Strategy**: Route forms (via ResNet clustering) to specialized checkpoints if each layout is drastically different.

------

## 3.3 Training Objectives

We employ a **dual-objective**: **Cross-Entropy (CE)** for text generation and **contrastive** learning for embedding robustness.

### 3.3.1 Cross-Entropy (CE) for JSON Generation

- **Purpose**: Teach the model to produce the correct JSON from a scanned image.
- Process:
  1. The model receives an image plus an instruction prompt, e.g., “Extract [name, policy_number, date_of_birth] as JSON.”
  2. It outputs tokens for the structured JSON, such as `{"name":"John Doe","policy_number":"AB123","date_of_birth":"1990-01-01"}`.
  3. We compute **token-level CE** against the ground truth JSON.

### 3.3.2 Contrastive Loss for Robust Embeddings

- **Purpose**: Encourage consistent embeddings for visually similar images (e.g., different scans or augmented versions of the same form).
- Positive vs. Negative Pairs:
  - **Positive**: Original image vs. augmented/distorted version.
  - **Negative**: Different forms or clearly different content.
- **Formula** (InfoNCE example):

![image-20250203160237085](C:\Users\YuxiangWang\AppData\Roaming\Typora\typora-user-images\image-20250203160237085.png)

where sim(⋅) is typically a dot-product or cosine similarity,  τ is a temperature, and N is the total number of samples in the contrastive batch.

### 3.3.3 Combined Objective

We combine both objectives with a weighting factor λ:

![image-20250203160358003](C:\Users\YuxiangWang\AppData\Roaming\Typora\typora-user-images\image-20250203160358003.png)

This ensures the model:

- Learns to generate accurate JSON (CE),
- Maintains robustness to distortions (contrastive).

------

## 3.4 Key Hyperparameters

| **Hyperparameter**              | **Typical Range** | **Notes**                                                    |
| ------------------------------- | ----------------- | ------------------------------------------------------------ |
| **Learning Rate**               | 1e-5 to 1e-4      | For LoRA-updated layers; tune based on data size & GPU budget |
| **Batch Size**                  | 32                | Balances memory usage & stable training                      |
| **Contrastive Weight (lambda)** | 0.1               | Ensures CE remains dominant; tweak if distortions are severe |
| **Temperature (tau)**           | 0.05              | In InfoNCE loss; lower = sharper distribution                |
| **Max Sequence Length**         | 2048 tokens       | Must include prompt + JSON                                   |
| **Epochs**                      | 5                 | Early stopping based on validation performance               |

------

## 3.5 Implementation Considerations

### 3.5.1 Layer Freezing & LoRA

- **Vision Encoder:**
  - Unfreeze top **2–4 layers** to adapt to handwriting.
  - Deeper layers remain frozen to retain general vision knowledge.
- **Projection Layer:**
  - Insert LoRA modules (rank ~8–16) to capture new domain-specific mapping (handwriting to text tokens).
- **Language Decoder:**
  - Typically kept frozen if domain text matches standard usage.
  - Optionally insert LoRA into cross-attention if domain language is highly specialized.

### 3.5.2 Training Loop Example

1. **Batch Sampling:**
   - Each batch has (image, JSON) for CE and (image, augmented_image) for contrastive.
2. **Forward Pass:**
   - Compute **CE loss** on text output.
   - Compute **contrastive loss** on the projection-layer embeddings.
   - Combine: ![image-20250203160519032](C:\Users\YuxiangWang\AppData\Roaming\Typora\typora-user-images\image-20250203160519032.png)
3. **Backprop & Optimize:**
   - Update only LoRA and unfrozen layers.
   - Use **AdamW** with moderate weight decay (e.g., 1e-4).
4. **Validation:**
   - Check field-level accuracy for JSON outputs.
   - Optionally measure embedding distances between positive pairs.

### 3.5.3 Handling JSON Output

- **Prompt Design:**
  - Use clear instructions to produce well-formed JSON.
  - Example: “`Please output a valid JSON with {field1, field2,...}.`”
- **Post-processing:**
  - Minimal, typically just verifying JSON structure.
  - If malformed, a simple parser can attempt corrections or prompt re-generation.

------

## 3.6 Example Code Snippet

```python
import torch
from torch.optim import AdamW

model = load_llava_model("LLaVA-1.6")
freeze_all_layers(model)
unfreeze_top_vision_layers(model, num_layers=4)
apply_lora_to_projection(model, rank=8)

optimizer = AdamW(model.parameters(), lr=1e-5)
contrastive_lambda = 0.1

def training_step(batch):
    images, json_labels, aug_images, neg_images = batch
    
    # Cross-Entropy (CE) loss for JSON
    ce_loss = model.forward_ce_loss(images, json_labels)
    
    # Contrastive loss
    pos_loss = model.forward_contrastive(images, aug_images, positive=True)
    neg_loss = model.forward_contrastive(images, neg_images, positive=False)
    contrastive_loss = pos_loss + neg_loss
    
    total_loss = ce_loss + contrastive_lambda * contrastive_loss
    total_loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss.item()
```

------

## 3.7 Summary

1. **Base Model**: LLaVA 1.6, chosen for robust image-to-text tasks.
2. **LoRA & Selective Unfreezing**: Minimizes memory usage while adapting to handwriting.
3. Dual-Objective:
   - **CE** secures accurate JSON output.
   - **Contrastive** ensures embeddings remain stable under distortions.
4. **Implementation**: Balanced data batching, partial layer freezing, and careful prompt design yield strong accuracy gains with lower manual verification.

By combining these techniques, we effectively handle handwritten forms in a resource-efficient way, delivering consistent JSON extractions across varying scan qualities and styles.



# 4. Training & Validation

> **Summary:**
>  This section outlines how we train our parameter-efficient Vision-LLM setup—covering data splitting, batch construction, optimization details, validation strategies, and metrics used to evaluate performance. Our goal is to ensure the final model can reliably extract JSON fields from handwritten forms while minimizing manual corrections.

------

## 4.1 Training Process

### 4.1.1 Data Splitting & Organization

- **Train / Validation / Test**
  - We create splits from each data source (real, synthetic, augmented).
  - Aim to preserve the proportion of each form type across splits to avoid data imbalance.
- **Stratification by Form Type**
  - If certain form layouts are rare, ensure they appear in both train and validation sets.
  - In some cases, we may hold out entire form types to test generalization on unseen layouts.

### 4.1.2 Batch Construction

- **Combined Batches**
  - Each batch typically contains:
    1. `(image, json_label)` pairs for **Cross-Entropy (CE)**.
    2. `(image_original, image_augmented)` (and possibly `(image, image_negative)`) pairs for **contrastive**.
- **Dynamic Sampling**
  - We randomly select some real vs. synthetic examples, ensuring a representative mix.
  - For contrastive, we sample positive pairs (same content, different distortion) and negative pairs (different forms).

### 4.1.3 Optimization & Scheduling

- **Optimizer**
  - **AdamW** with weight decay (e.g., `1e-4`).
  - Learning rate ~`1e-5` for LoRA layers.
- **Learning Rate Schedulers**
  - A linear warmup (5–10% of total steps) to stabilize early training, followed by a gradual decay.
- **Epochs & Early Stopping**
  - Typically 5–15 epochs.
  - Monitor a validation metric (e.g., field-level accuracy) to trigger early stopping if performance plateaus or degrades.

### 4.1.4 Loss Computation

- **Dual-Objective**

  ![image-20250203161010482](C:\Users\YuxiangWang\AppData\Roaming\Typora\typora-user-images\image-20250203161010482.png)

  **CE** is computed by comparing predicted JSON tokens to ground-truth tokens.

  **Contrastive** uses InfoNCE or a similar approach to align embeddings for positive pairs and separate embeddings for negatives.

- **Balancing (λ)**

  - Start with a small value (e.g., 0.05) so that JSON accuracy remains the primary objective.
  - Increase if see persistent issues with image distortion or layout confusion.

------

## 4.2 Validation Approach

### 4.2.1 Validation Set Composition

- **Real vs. Synthetic**
  - Include both real scanned forms (to measure real-world performance) and synthetic forms (to assess coverage of rare fields).
- **Per-Form Analysis**
  - Evaluate each form type separately if their complexity varies (e.g., “Form A Page 1” vs. “Form B Page 2”).
  - This reveals whether certain layouts remain problematic.

### 4.2.2 Inference Procedure

1. **Image Preprocessing**: Same normalization steps used in training (resizing, cropping).
2. **Prompting**: Provide the same instruction style (e.g., “`Extract the following fields as JSON:`”) used during training.
3. **JSON Parsing:**
   - The model outputs a text string, which we parse as JSON.
   - If parsing fails, we note it as an error or apply minimal corrections (e.g., fix unbalanced braces).

### 4.2.3 Metrics & Evaluation

1. **Field-Level Accuracy**

   - Compare each key in the output JSON to the corresponding ground-truth value.
   - Tolerate small differences in whitespace or punctuation, but require matching strings for critical fields (e.g., policy number).

2. **Exact JSON Match**

   - A stricter measure: the entire generated JSON (structure + contents) must match ground-truth.
   - Useful for tasks where the shape of the JSON is fixed.

3. **Word/Character Error Rate (WER/CER)**

   - Borrowed from OCR evaluation.
   - Especially informative for fields like names or addresses with high variability.

4. **Confidence or Probability Scores**

   (Optional)

   - Some systems can output confidence estimates for each token or field.
   - Low-confidence areas might then be flagged for human review.

### 4.2.4 Example Validation Metrics

- **Precision/Recall on Key Fields:**
  - For checkboxes or yes/no fields, measure classification performance.
- **Time-to-Review:**
  - How long do human annotators take to correct errors? If it decreases after model improvements, it indicates higher practical value.

------

## 4.3 Common Pitfalls & Troubleshooting

- **Overfitting on Synthetic Data**
  - Risk: The model becomes overly reliant on synthetic examples, which may not capture true handwriting complexity.
  - Mitigation: Ensure enough real data in each batch and use data augmentation for real scans.
- **Malformed JSON Outputs**
  - Risk: The model occasionally produces syntax errors (missing brackets, quotes).
  - Mitigation: Post-process with a JSON validator or re-prompt for corrections.
- **Contrastive Loss Imbalance**
  - If λ is too high, the model might overfocus on embedding alignment and degrade text generation quality.
  - Balance carefully and monitor CE vs. contrastive performance metrics.
- **Scaling to Multiple Form Types**
  - Different forms might require different fields or drastically different structures.
  - Use consistent prompts or multi-checkpoint routing if forms diverge extensively.

------

## 4.4 Example Validation Run

1. **Data Setup:**
   - Validation dataset of ~500 real scanned forms + 500 synthetic forms.
2. **Execution:**
   - Inference with batch size = 4 (due to GPU constraints).
   - Model outputs JSON for each form.
3. **Results:**
   - **Field-Level Accuracy**: 98.9% for real scans, 99.7% for synthetic.
   - **Exact JSON Match**: 99% on real data, 99% on synthetic.
   - **Manual Review Reduction**: The average time spent by annotators dropped by 70% relative to the legacy OCR pipeline.

------

## 4.5 Summary

1. **Training** involves a carefully balanced dual-objective (CE + contrastive) approach, selective data batching, and parameter-efficient updates (LoRA).
2. **Validation** hinges on field-level accuracy metrics and occasionally stricter JSON-match measures, ensuring the system truly reduces manual correction needs.
3. **Best Practices**: Maintain a healthy balance of real and synthetic data, keep an eye on malformed JSON, and fine-tune the contrastive weight to avoid overshadowing JSON generation accuracy.

By following these principles, the resulting Vision-LLM model consistently extracts high-fidelity structured data from challenging hand-written scans, paving the way for a more reliable and cost-effective document workflow.



# 5. Deployment & Infrastructure

> **Summary:**
>  In this phase, we package our Vision-LLM solution into Docker containers and deploy it on AWS infrastructure using Kubernetes (K8s). This ensures a scalable, reliable, and maintainable environment for inference. Below, we outline the key components, architectural choices, and operational considerations.

------

## 5.1 Containerization & Environment

### 5.1.1 Docker Image

- Base Image
  - Typically built on a Linux distribution with GPU support (e.g., `nvidia/cuda` base image) if GPU inference is required.
  - Install necessary libraries: PyTorch, Transformers, LoRA dependencies, and any custom code.
- Entry Point
  - A lightweight Python server (e.g., FastAPI, Flask)
  - Exposes REST or gRPC endpoints for image upload and JSON response.

```bash
# Example Dockerfile snippet
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY . /app
CMD ["python3", "serve.py"]  # Example entry point
```

> **Tip:** Keep the Docker image minimal to reduce deployment overhead and improve startup times.

### 5.1.2 Environment Variables & Configuration

- AWS Credentials

   (if reading from S3)

  - Stored securely in AWS Secrets Manager or environment variables.

- Model Checkpoint

  - Either baked into the Docker image or fetched from S3 / ECR on container startup.

- ResNet Classifier & LoRA Model Weights

  - Ensure both the form-classification model (ResNet) and the VLLM (LLaVA + LoRA layers) are accessible at run-time.

------

## 5.2 Kubernetes on AWS

### 5.2.1 Cluster Setup

- AWS EKS (Elastic Kubernetes Service)
  - Managed Kubernetes control plane on AWS.
  - Nodes can be GPU-backed instances (e.g., `p2`, `p3`, or `g4dn`) for efficient inference on large models.
- Node Autoscaling
  - Configure Cluster Autoscaler or Karpenter to scale the number of GPU nodes based on CPU/GPU utilization and queue lengths.

### 5.2.2 Service & Networking

- Service Type
  - Typically a LoadBalancer service type in Kubernetes to expose the model inference endpoint externally.
  - For internal use, a ClusterIP service suffices.
- Ingress Controller
  - An NGINX or AWS ALB (Application Load Balancer) ingress can route traffic to the correct service.
  - SSL termination can be handled at the ALB or the ingress layer.

### 5.2.3 Deployment Manifests

- Deployment

  - A `Deployment` spec in K8s defines the desired replica count and container images.

  - Example snippet:

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-form-extraction
    spec:
      replicas: 2
      selector:
        matchLabels:
          app: vllm-form-extraction
      template:
        metadata:
          labels:
            app: vllm-form-extraction
        spec:
          containers:
          - name: form-extractor
            image: <ECR_REPO_URI>/vllm-form-extraction:latest
            resources:
              limits:
                nvidia.com/gpu: 1
    ```

- ConfigMap / Secrets

  - Store environment variables, model references, or AWS credentials.
  - Ensures minimal changes to the container image for environment-specific parameters.

------

## 5.3 Inference Pipeline

### 5.3.1 Request Flow

1. **Input**: A scanned form image (PNG/JPEG).
2. **ResNet Classification**: A lightweight microservice call that identifies the form type (e.g., “Form A Page 1”).
3. **VLLM Inference**: The correct specialized (or single multi-type) checkpoint is loaded, the image is passed into the model, and JSON is generated.
4. **Post-Processing**: A minimal parser validates the JSON and corrects any minor syntax issues if needed.

> **Note**: This pipeline can be orchestrated within the same container or split into separate microservices for modular scaling.

### 5.3.2 Latency & Scaling Considerations

- GPU vs. CPU
  - For large VLLMs, GPU inference drastically reduces latency but increases cost.
  - CPU-based pods might suffice for lower loads or smaller models.
- Autoscaling
  - Horizontal Pod Autoscaler (HPA) can scale the number of inference pods based on metrics like average CPU/GPU utilization or request queue length.
- Caching & Warm Start
  - Preload model weights at container startup to avoid on-demand loading overhead.
  - Keep a small number of pods “warm” during off-peak hours to handle bursts.

------

## 5.4 Supporting Services & Storage

### 5.4.1 Object Storage (S3)

- Form Images
  - Incoming images can be stored in S3 prior to processing.
  - Processed results (JSON) can also be archived in S3.
- Model Artifacts
  - LoRA weights and baseline model checkpoints stored in S3 or ECR.
  - Containers fetch them during startup if not baked in.

### 5.4.2 Monitoring & Logging

- CloudWatch / Prometheus
  - Log GPU utilization, memory usage, and custom application metrics (e.g., average inference time).
- Centralized Logging
  - Collect container logs in CloudWatch Logs or Elasticsearch to track failures and debug issues.
- Alerting
  - Set alarms on important metrics (e.g., if inference latency spikes or if GPU utilization is near 100% for extended periods).

### 5.4.3 CI/CD Integration

- Build Pipeline
  - Jenkins to build Docker images and push to **ECR**.
- Deploy Pipeline
  - Automated deploys to EKS with rolling updates.
  - Staged environments (dev, staging, prod) to test new model versions.

------

## 5.5 Security & Compliance

- Network Isolation
  - Restrict inbound/outbound traffic using Kubernetes Network Policies and Security Groups.
  - Keep the model pods in private subnets if external access isn’t required.
- Data Encryption
  - Encrypt data at rest in S3.
  - Enforce TLS in transit for requests to the inference endpoint.
- IAM Roles
  - Pods assume least-privilege IAM roles to access only the resources they need (model artifacts, S3 buckets).

------

## 5.6 Cost Optimization

- Spot Instances
  - Consider spot-priced GPU instances if workload can tolerate interruptions.
- Right-Sizing
  - Monitor usage to ensure no over-provisioning of GPU memory or vCPUs.
- Autoscaling
  - Scale down to minimal replicas during off-peak hours.

------

## 5.7 Summary

1. **Containerized Deployment**: Docker images with GPU support are deployed via AWS EKS for easy scaling and management.
2. **Microservice Architecture**: A classification step (ResNet) followed by VLLM inference ensures form-type–specific handling.
3. **Scalability & Monitoring**: K8s autoscalers, CloudWatch/Prometheus metrics, and robust logging ensure smooth operations under varying loads.
4. **Security & Compliance**: Encrypted data in S3, TLS endpoints, IAM-based access control, and network isolation meet enterprise security needs.

By leveraging Kubernetes on AWS, we achieve a flexible infrastructure capable of handling high-throughput inference while maintaining the cost efficiency and reliability required for production environments.

# 6. Performance Results & Lessons Learned

> **Summary:**
>  After training and deploying our Vision-LLM solution, we evaluated its real-world performance on internal validation sets and production-like test scenarios. This section provides a concise overview of metrics achieved, as well as key challenges encountered and how they were resolved.

------

## 6.1 Metrics Summary

1. **Field-Level Accuracy**
   - **Real Scanned Forms**: Achieved **99.5%** accuracy across critical fields (e.g., policy number, customer name) in our final validation set.
   - **Synthetic & Augmented Forms**: Slightly higher accuracy (**99.9%**) due to cleaner data and perfectly labeled ground truth.
2. **Exact JSON Match**
   - **Strict Format Checks**: **98.9%** of outputs exactly matched the ground-truth JSON (including punctuation and structure) for real scans.
   - **Common Discrepancies**: Minor bracket or quote placement errors for ~0.5% of the remaining cases, mostly fixable by simple post-processing.
3. **Manual Verification Time**
   - **Reduction**: Manual review time dropped by **70%**, indicating less frequent or less severe corrections compared to the legacy OCR pipeline.
   - **Operational Impact**: Estimated annual cost savings through reduced labor and faster turnaround times.
4. **Inference Latency**
   - **GPU-Based**: ~3000–8000 ms per form (batch size 2–4) for the VLLM.
5. **Contrastive vs. Non-Contrastive Ablation**
   - Models trained with **Contrastive + CE** exhibited about **3–5%** higher field-level accuracy on heavily distorted forms compared to CE-only models.
   - Validated that embedding alignment effectively mitigates scanning artifacts.

------

## 6.2 Key Challenges & Solutions

1. **Handwriting Variability**
   - **Challenge**: Drastically different penmanship styles resulted in inconsistent recognition.
   - Solution:
     - Leveraged **augmented real images** (e.g., random distortion, blur) and **positive/negative contrastive pairs** to teach the model robust feature extraction.
     - Unfreezing top layers in the vision encoder via LoRA allowed domain adaptation without fully retraining.
2. **Limited High-Quality Labels**
   - **Challenge**: Few thoroughly annotated real forms, which risked underfitting or overfitting to trivial patterns.
   - Solution:
     - **Synthetic Data Generation** using an LLM to produce realistic sample forms at scale.
     - **Hybrid Training**: Interleaving synthetic forms with real scans to maintain realism.
3. **Malformed JSON Output**
   - **Challenge**: Occasional bracket or punctuation errors caused invalid JSON, complicating downstream parsing.
   - Solution:
     - **Prompt Refinement**: Repeatedly instruct the model to produce strictly valid JSON, with added examples.
     - **Simple Post-Processing**: A script to detect and correct minor bracket/quote imbalances.
4. **Balancing CE & Contrastive Loss**
   - **Challenge**: If contrastive weight (λ) was too large, JSON text generation accuracy dipped.
   - Solution:
     - Began training with a small λ(0.1). Gradually increased to 0.15–0.2 while monitoring JSON accuracy.
     - Confirmed final ratio optimized both text fidelity and embedding robustness.
5. **Scaling Inference on AWS**
   - **Challenge**: Handling surges in request volume while keeping latency low.
   - Solution:
     - **K8s Autoscaling** (on GPU nodes) with AWS EKS.
     - **Caching Model Weights** on node local storage, ensuring quick container restarts and minimal model load overhead.

------

> **Takeaway:** Through careful curation of data (real + synthetic + augmented), a hybrid CE + contrastive training objective, and parameter-efficient fine-tuning with LoRA, we achieved a notable accuracy improvement over the legacy OCR approach. Real-world feedback suggests fewer escalations, faster processing, and meaningful cost savings in manual review.



### 3.7 Future Work & Extensions

> **Summary:**
>  Although our current setup has yielded significant improvements, there are additional directions to explore for enhanced accuracy, scalability, and maintainability. Below, we outline several key next steps and potential improvements, with a particular focus on unifying multiple form types into a single Vision-LLM using special tokens.

------

#### Next Steps

1. **Unified Model for All Form Types**
   - **Motivation**: Instead of maintaining multiple checkpoints (routing forms to specialized models), a single model handles all form variations.
   - Method:
     - **Special Token**: Prepend a token to the text prompt indicating the form type, e.g., `[FORM_A_PAGE_1]`, `[FORM_B_PAGE_2]`, etc.
     - **Training Data**: Mix data from all form types in a single training set, ensuring balanced coverage of each layout.
   - **Benefit**: Reduces deployment complexity and memory overhead by consolidating multiple specialized models into one.
2. **Automatic Schema Detection**
   - **Current Approach**: We rely on ResNet-based clustering to identify form types and choose the corresponding schema.
   - **Future Upgrade**: Train the Vision-LLM to dynamically infer the schema from visual cues (e.g., key field labels). This removes the need for external classification.
3. **Multi-Task Extraction**
   - **Expansion**: Beyond extracting text fields, the model could handle *structured fields* like checkboxes, signatures, or tables within certain forms.
   - **Technical Requirement**: Additional training data with bounding-box or segmentation labels to help the model learn more granular visual features.

------

#### Potential Improvements

1. **More Robust Data Augmentation**
   - **Goal**: Tackle extreme scan conditions (heavy glare, partial occlusion, folds).
   - Techniques:
     - Advanced synthetic warping, style-transfer approaches that mimic different pen strokes or cursive handwriting.
2. **Adaptive Inference**
   - **On-Device Distillation**: For edge or offline scenarios, a smaller distilled version of the Vision-LLM could run locally, reducing latency and cloud dependency.
   - **Confidence Thresholds**: Output a confidence score per field. If below a threshold, escalate to human review for critical fields.
3. **Schema Validation Layer**
   - **Concept**: Post-process the generated JSON by comparing it to a known schema (e.g., required keys, data types).
   - Implementation:
     - A simple rule-based or function-based validator that checks mandatory fields (e.g., “date_of_birth” must be a valid date) before finalizing output.
4. **Cross-Form Transfer Learning**
   - **Aim**: Rapidly adapt the model to new form types with minimal labeled data.
   - **Approach**: LoRA-based incremental fine-tuning or “prompt-tuning” for newly introduced fields while retaining prior knowledge.

------

> **Overall**: The single-model approach with special form-type tokens stands out as a pivotal enhancement. It streamlines maintenance and potentially yields better generalization, given the model sees all form types during training. Coupled with advanced data augmentation, RLHF, and schema validation, these future directions can push the boundaries of automated form extraction—resulting in higher accuracy, broader coverage of form variations, and further reductions in manual verification.