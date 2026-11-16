args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: Rscript metafor_robustness.R <input.csv> <output.json> <model> <effect_type> <pooling_scale>")
}

input_path <- args[[1]]
output_path <- args[[2]]
primary_model <- args[[3]]
effect_type <- args[[4]]
pooling_scale <- args[[5]]

if (!requireNamespace("metafor", quietly = TRUE)) {
  stop("R package 'metafor' is not installed. Run install.packages('metafor').")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("R package 'jsonlite' is not installed. Run install.packages('jsonlite').")
}

dat <- read.csv(input_path, stringsAsFactors = FALSE)
# Preserve the executed main method, including its CI procedure.
primary_method <- if (primary_model == "fixed_effect") "FE" else if (
  grepl("__reml", pooling_scale, fixed=TRUE)) "REML" else if (
  grepl("__paule_mandel", pooling_scale, fixed=TRUE)) "PM" else "DL"
primary_test <- if (primary_model == "random_effects" &&
  grepl("__hartung_knapp_t", pooling_scale, fixed=TRUE)) "knha" else "z"

safe_value <- function(expr) {
  tryCatch(expr, error = function(e) NULL)
}
safe_number <- function(x) {
  if (length(x) == 0 || is.null(x) || is.na(x[[1]]) || !is.finite(x[[1]])) return(NULL)
  unname(as.numeric(x[[1]]))
}
fit_summary <- function(fit) {
  if (is.null(fit)) return(NULL)
  list(
    pooled_log_effect = safe_number(fit$b),
    ci_lower_log = safe_number(fit$ci.lb),
    ci_upper_log = safe_number(fit$ci.ub),
    i_squared = safe_number(fit$I2),
    tau_squared = safe_number(fit$tau2),
    q = safe_number(fit$QE),
    q_p_value = safe_number(fit$QEp)
  )
}

fit_primary <- metafor::rma.uni(yi = dat$yi, vi = dat$vi, method = primary_method, test = primary_test)
fit_fixed <- safe_value(metafor::rma.uni(yi = dat$yi, vi = dat$vi, method = "FE"))
fit_random <- safe_value(metafor::rma.uni(yi = dat$yi, vi = dat$vi, method = "DL"))

loo_df <- safe_value(as.data.frame(metafor::leave1out(fit_primary)))
leave_one_out <- if (is.null(loo_df)) list() else lapply(seq_len(nrow(loo_df)), function(i) {
  list(
    study_id = dat$study_id[[i]],
    evidence_id = dat$evidence_id[[i]],
    pooled_log_effect = safe_number(loo_df$estimate[[i]]),
    ci_lower_log = safe_number(loo_df$ci.lb[[i]]),
    ci_upper_log = safe_number(loo_df$ci.ub[[i]]),
    i_squared = safe_number(loo_df$I2[[i]]),
    tau_squared = safe_number(loo_df$tau2[[i]])
  )
})

cook <- safe_value(metafor::cooks.distance(fit_primary))
rstud <- safe_value(as.data.frame(metafor::rstudent(fit_primary)))
influence <- lapply(seq_len(nrow(dat)), function(i) {
  list(
    study_id = dat$study_id[[i]],
    evidence_id = dat$evidence_id[[i]],
    cook_distance = if (is.null(cook)) NULL else safe_number(cook[[i]]),
    studentized_residual = if (is.null(rstud)) NULL else safe_number(rstud$z[[i]])
  )
})

baujat_df <- safe_value(as.data.frame(metafor::baujat(fit_primary)))
baujat <- if (is.null(baujat_df)) list() else lapply(seq_len(nrow(baujat_df)), function(i) {
  list(
    study_id = dat$study_id[[i]],
    evidence_id = dat$evidence_id[[i]],
    heterogeneity_contribution = safe_number(baujat_df$x[[i]]),
    influence_on_pooled_effect = safe_number(baujat_df$y[[i]])
  )
})

result <- list(
  k = nrow(dat),
  independent_study_count = length(unique(dat$study_id)),
  effect_type = effect_type,
  primary_model = primary_model,
  primary_method = primary_method,
  primary_ci = primary_test,
  model_fits = list(
    primary = fit_summary(fit_primary),
    fixed_effects = fit_summary(fit_fixed),
    random_effects = fit_summary(fit_random)
  ),
  leave_one_out = leave_one_out,
  influence = influence,
  baujat = baujat
)
jsonlite::write_json(result, output_path, auto_unbox = TRUE, null = "null", pretty = TRUE, digits = 16)
