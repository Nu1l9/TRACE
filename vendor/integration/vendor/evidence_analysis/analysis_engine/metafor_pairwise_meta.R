args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop("Usage: Rscript metafor_pairwise_meta.R <input.csv> <output.json> <model> <effect_scale> <effect_type> <pooling_scale>")
}

input_path <- args[[1]]
output_path <- args[[2]]
pooling_model <- args[[3]]
effect_scale <- args[[4]]
effect_type <- args[[5]]
pooling_scale <- args[[6]]

if (!requireNamespace("metafor", quietly = TRUE)) stop("R package 'metafor' is not installed.")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("R package 'jsonlite' is not installed.")

dat <- read.csv(input_path, stringsAsFactors = FALSE)
safe_number <- function(value) {
  if (length(value) == 0 || is.null(value) || is.na(value[[1]]) || !is.finite(value[[1]])) return(NULL)
  unname(as.numeric(value[[1]]))
}

rho_marker <- regmatches(pooling_scale, regexpr("__clustered_rho_0_[369]", pooling_scale))
if (length(rho_marker) && nzchar(rho_marker)) {
  rho <- as.numeric(sub("__clustered_rho_0_", "0.", rho_marker, fixed = TRUE))
  if (pooling_model != "random_effects" || effect_scale != "log_ratio") {
    stop("Clustered synthesis requires random-effects log-ratio inputs.")
  }
  if (anyDuplicated(dat$evidence_id) || any(!is.finite(dat$vi)) || any(dat$vi <= 0)) {
    stop("Clustered synthesis requires unique evidence IDs and positive finite variances.")
  }
  if (length(unique(dat$study_id)) < 3) {
    stop("Cluster-robust inference requires at least three independent studies.")
  }
  # Source papers do not provide cross-outcome sampling covariances. The
  # declared working correlation is explicit and is applied only within a
  # study; the confidence interval clusters at the independent-study level.
  covariance <- metafor::vcalc(
    vi = vi, cluster = study_id, obs = evidence_id,
    data = dat, rho = rho
  )
  fit <- metafor::rma.mv(
    yi = dat$yi, V = covariance,
    random = ~ 1 | study_id/evidence_id, data = dat, method = "REML"
  )
  robust_fit <- metafor::robust(fit, cluster = dat$study_id, adjust = TRUE)
  result <- list(
    k = nrow(dat), independent_study_count = length(unique(dat$study_id)),
    effect_type = effect_type,
    model = paste0("multilevel_random_effects_cluster_robust_rho_", sub("^0\\.", "0_", as.character(rho))),
    pooled_effect = safe_number(exp(robust_fit$b)),
    ci_lower = safe_number(exp(robust_fit$ci.lb)),
    ci_upper = safe_number(exp(robust_fit$ci.ub)),
    # A univariate I-squared or tau-squared is not comparable to the two
    # variance components of this multilevel model.
    i_squared = NULL,
    q = safe_number(fit$QE),
    q_p_value = safe_number(fit$QEp),
    tau_squared = NULL
  )
  jsonlite::write_json(result, output_path, auto_unbox = TRUE, null = "null", pretty = TRUE, digits = NA)
  quit(save = "no")
}

is_mh_or <- pooling_model == "fixed_effect" &&
  grepl("mantel_haenszel", pooling_scale, fixed = TRUE) &&
  effect_scale == "log_ratio" && effect_type == "OR" &&
  all(stats::complete.cases(dat[, c("ai", "bi", "ci", "di")]))

selected_model <- pooling_model
if (is_mh_or) {
  fit <- metafor::rma.mh(
    measure = "OR", ai = dat$ai, bi = dat$bi, ci = dat$ci, di = dat$di,
    add = 0.5, to = "only0"
  )
} else {
  if (grepl("__heterogeneity_switch", pooling_scale, fixed = TRUE)) {
    fixed_fit <- metafor::rma.uni(yi = dat$yi, vi = dat$vi, method = "FE")
    use_random <- !is.null(fixed_fit$QEp) && !is.null(fixed_fit$I2) &&
      (fixed_fit$QEp < 0.10 || fixed_fit$I2 > 50)
    selected_model <- if (use_random) "random_effects" else "fixed_effect"
  }
  method <- if (selected_model == "fixed_effect") "FE" else if (
    grepl("__reml", pooling_scale, fixed = TRUE)
  ) "REML" else if (
    grepl("__paule_mandel", pooling_scale, fixed = TRUE)
  ) "PM" else "DL"
  # Hartung-Knapp uses the t distribution and adjusts the random-effects
  # standard error. It is activated only when the public profile declares it.
  test_method <- if (
    selected_model == "random_effects" &&
    grepl("__hartung_knapp_t", pooling_scale, fixed = TRUE)
  ) "knha" else "z"
  fit <- metafor::rma.uni(yi = dat$yi, vi = dat$vi, method = method, test = test_method)
}

if (effect_scale == "log_ratio") {
  pooled_effect <- safe_number(exp(fit$b))
  ci_lower <- safe_number(exp(fit$ci.lb))
  ci_upper <- safe_number(exp(fit$ci.ub))
} else if (grepl("^logit_(sensitivity|specificity)", pooling_scale)) {
  pooled_effect <- safe_number(stats::plogis(fit$b))
  ci_lower <- safe_number(stats::plogis(fit$ci.lb))
  ci_upper <- safe_number(stats::plogis(fit$ci.ub))
} else if (grepl("^freeman_tukey_proportion", pooling_scale)) {
  pooled_effect <- safe_number(sin(fit$b / 2)^2)
  ci_lower <- safe_number(sin(fit$ci.lb / 2)^2)
  ci_upper <- safe_number(sin(fit$ci.ub / 2)^2)
} else {
  pooled_effect <- safe_number(fit$b)
  ci_lower <- safe_number(fit$ci.lb)
  ci_upper <- safe_number(fit$ci.ub)
}

result <- list(
  k = nrow(dat),
  effect_type = effect_type,
  model = selected_model,
  pooled_effect = pooled_effect,
  ci_lower = ci_lower,
  ci_upper = ci_upper,
  i_squared = safe_number(fit$I2),
  q = safe_number(fit$QE),
  q_p_value = safe_number(fit$QEp),
  tau_squared = safe_number(fit$tau2)
)
jsonlite::write_json(result, output_path, auto_unbox = TRUE, null = "null", pretty = TRUE, digits = NA)
