args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(4, 6))) {
  stop("Usage: Rscript metafor_reporting_bias.R <input.csv> <output.json> <model> <effect_type> [funnel.svg contour_funnel.svg]")
}

input_path <- args[[1]]
output_path <- args[[2]]
pooling_model <- args[[3]]
effect_type <- args[[4]]
funnel_path <- if (length(args) == 6) args[[5]] else NULL
contour_funnel_path <- if (length(args) == 6) args[[6]] else NULL

if (!requireNamespace("metafor", quietly = TRUE)) {
  stop("R package 'metafor' is not installed. Run install.packages('metafor').")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("R package 'jsonlite' is not installed. Run install.packages('jsonlite').")
}

dat <- read.csv(input_path, stringsAsFactors = FALSE)
method <- if (pooling_model == "random_effects") "REML" else "FE"
fit <- metafor::rma.uni(yi = dat$yi, vi = dat$vi, method = method)

draw_svg <- function(path, draw) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  grDevices::svg(filename = path, width = 6.4, height = 5.0, pointsize = 9)
  on.exit(grDevices::dev.off(), add = TRUE)
  draw()
}

if (!is.null(funnel_path) && !is.null(contour_funnel_path)) {
  draw_svg(funnel_path, function() {
    metafor::funnel(
      fit,
      yaxis = "sei",
      atransf = exp,
      xlab = paste0(effect_type, " (ratio scale)"),
      ylab = "Standard error",
      col = "#21688f",
      bg = "#21688f",
      refline = as.numeric(fit$b),
      lty = 2
    )
    graphics::title("Funnel plot")
  })
  draw_svg(contour_funnel_path, function() {
    metafor::funnel(
      fit,
      yaxis = "sei",
      atransf = exp,
      xlab = paste0(effect_type, " (ratio scale)"),
      ylab = "Standard error",
      level = c(90, 95, 99),
      shade = c("#ffffff", "#dbe9ef", "#9fc4d6"),
      refline = 0,
      lty = 2,
      col = "#21688f",
      bg = "#21688f",
      legend = list(x = "bottomright", show = "pvals", bty = "n")
    )
    graphics::abline(v = as.numeric(fit$b), lty = 3, lwd = 1.1, col = "#4d4d4d")
    graphics::title("Contour-enhanced funnel plot")
  })
}

safe_value <- function(expr) {
  tryCatch(expr, error = function(e) NULL)
}
scalar_number <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(as.numeric(x)[[1]]) || !is.finite(as.numeric(x)[[1]])) return(NULL)
  unname(as.numeric(x)[[1]])
}

reg <- safe_value(metafor::regtest(fit, predictor = "sei"))
rank <- safe_value(metafor::ranktest(fit))
taf <- safe_value(metafor::trimfill(fit))
reg_statistic <- if (is.null(reg)) NULL else if (!is.null(reg$zval)) unname(reg$zval) else unname(reg$tval)
fit_summary <- list(
  pooled_log_effect = scalar_number(fit$b),
  ci_lower_log = scalar_number(fit$ci.lb),
  ci_upper_log = scalar_number(fit$ci.ub)
)

result <- list(
  k = nrow(dat),
  independent_study_count = length(unique(dat$study_id)),
  effect_type = effect_type,
  model = pooling_model,
  unadjusted = fit_summary,
  regtest = if (is.null(reg)) NULL else list(
    statistic = reg_statistic,
    p_value = unname(reg$pval),
    limit_estimate = unname(reg$beta)
  ),
  ranktest = if (is.null(rank)) NULL else list(
    kendall_tau = unname(rank$tau),
    p_value = unname(rank$pval)
  ),
  trimfill = if (is.null(taf)) NULL else list(
    imputed_studies = scalar_number(taf$k0),
    side = taf$side,
    adjusted_log_effect = scalar_number(taf$b),
    adjusted_ci_lower_log = scalar_number(taf$ci.lb),
    adjusted_ci_upper_log = scalar_number(taf$ci.ub)
  )
)

jsonlite::write_json(result, output_path, auto_unbox = TRUE, null = "null", pretty = TRUE)
