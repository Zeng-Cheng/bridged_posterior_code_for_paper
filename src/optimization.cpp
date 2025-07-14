#include <iostream>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// // [[Rcpp::export]]
// void power_method(const arma::mat& A, 
//   arma::vec& x0, 
//   double& lambda0, 
//   double tol = 1E-8,int max_iter=1000)
// {
//     int n = A.n_rows;
//     x0 = arma::randn(n);
//     x0 /= arma::norm(x0);
//     lambda0 = 0;
//     int iter = 0;
//     while (iter < max_iter)
//     {
//         arma::vec y = A*x0;
//         double lambda1 = arma::dot(y, x0) / arma::dot(x0, x0);
//         arma::vec x1 = y / arma::norm(y);
//         double rel_error = arma::norm(x1 - x0) / arma::norm(x1);
//         x0 = x1;
//         lambda0 = lambda1;
//         iter++;
//         if (rel_error < tol) break;
//     }
// }


// // [[Rcpp::export]]
// arma::mat lanczos(const arma::mat A, int m, double tol= 1E-6)
// {
//     int n = A.n_rows;
//     arma::vec alpha(m);
//     arma::vec beta(m);
//     arma::mat V(n, m);

//     arma::vec q = arma::randn(n);
//     V.col(0) = q / arma::norm(q);
//     beta(0) = 0;

//     for (int j = 0; j < m-1; j++)
//     {
//         arma::vec w = A*V.col(j);
//         alpha(j) = arma::dot(V.col(j), w);

//         if (j == 0)
//         {
//             w -= alpha(j)*V.col(j);
//         }else{
//             w -= alpha(j)*V.col(j) + beta(j)*V.col(j-1);
//         }

//         beta(j+1) = arma::norm(w);

//         // if (beta(j) < tol) break;
//         V.col(j+1) = w / beta(j+1);
//     }
//     alpha(m-1) = arma::dot(V.col(m-1), A*V.col(m-1));

//     // arma::mat eigvec;
//     // arma::vec eigval;
//     // arma::eig_sym(eigval, eigvec, V.t()*A*V);
//     return V;
// }


// Define the loss function to minimize
// [[Rcpp::export]]
double lossFunction(arma::mat L, arma::mat L_A, arma::mat ZW, double rho, double eta) {

    arma::mat diff0 = L - L_A;
    arma::mat diff1 = L - ZW;

    arma::uvec lower_indices =  arma::trimatl_ind(arma::size(L), -1);
    arma::vec L_lower_part = L(lower_indices);

    if (any(L_lower_part > 0)) {
        return arma::datum::inf;
    }

    double cost = arma::accu(diff0 % diff0) / 2.0;
    cost += arma::accu(diff1 % diff1) / 2.0 * eta;
    cost -= 2.0 * arma::accu(arma::log(-L_lower_part)) * rho;
    return cost;
}


// [[Rcpp::export]]
// Define the gradient function
arma::mat gradientFunction(arma::mat L, arma::mat L_A, arma::mat ZW, double rho, double eta) {

    arma::mat diff0 = L - L_A;
    arma::mat diff1 = L - ZW;
    
    arma::mat one_over_L = -1.0 / L;
    one_over_L.diag().zeros();

    arma::mat gradient_step0 = diff0 + diff1 * eta + rho * one_over_L;
    
    arma::vec gs0_diag = gradient_step0.diag();
    arma::mat gradient = arma::zeros(arma::size(L));

    gradient.each_row() -= gs0_diag.t();
    gradient.each_col() -= gs0_diag;

    gradient += 2 * gradient_step0;
    gradient += arma::diagmat(arma::sum(-gradient, 1));

    return gradient;
}

// Define the gradient descent algorithm with backtracking line search
// [[Rcpp::export]]
arma::mat gradientDescent(arma::mat L, arma::mat L_A, arma::mat ZW, double rho, double eta,
    double alpha, double beta, int maxGradientDescentIterations, double tol) {

    for (int i = 0; i < maxGradientDescentIterations; ++i) {
        double cost = lossFunction(L, L_A, ZW, rho, eta);
        arma::mat gradient = gradientFunction(L, L_A, ZW, rho, eta); 
        double step = 1;

        arma::mat gradient_lowtri = arma::trimatl(gradient);
        double gradient_norm_sqr = arma::accu(gradient_lowtri % gradient_lowtri);

        if (sqrt(gradient_norm_sqr) < tol) {
            break;
        }

        arma::mat param_new = L - step * gradient;
        while (lossFunction(param_new, L_A, ZW, rho, eta) > cost - alpha * step * gradient_norm_sqr) {
            step *= beta;
            param_new = L - step * gradient;
        }
        // Rcpp::Rcout << step << gradient_norm_sqr << std::endl;
        L -= step * gradient;
    }
    return L;
}


// Define the central path algorithm
//[[Rcpp::export]]
arma::mat centralPath(arma::mat L, arma::mat L_A, arma::mat ZW, double eta, 
    double alpha = 0.2, double beta = 0.6, int maxGradientDescentIterations=100,
    double tol = 1E-4, double target_rho = 0.001, double start_rho = 1E-2) {

    double rho = start_rho;
    while (rho >= target_rho) {
        L = gradientDescent(L, L_A, ZW, rho, eta,
            alpha, beta, maxGradientDescentIterations, tol);
        rho *= 0.3;
    }
    return L;
}

// Define the proximal mapping with respect to nuclear norm
//[[Rcpp::export]]
arma::mat nuclearNormProx(arma::mat X, double lambda){

    arma::mat U;
    arma::vec s;
    arma::mat V;

    arma::svd(U, s, V, X);
    s = (s - lambda) % (s > lambda);
    return  U * diagmat(s) * V.t();
}


// Define the ADMM algorithm
//[[Rcpp::export]]
arma::mat ADMM(arma::mat L, arma::mat L_A, double lambda, double eta = 2.0,
    int maxADMMIterations = 100, double alpha = 0.002, double beta = 0.6,
    int maxGradientDescentIterations = 20, double tol = 0.001,
    double target_rho = 1E-5, double tol_total_loss = 0.001,
    double start_rho = 0.1){

    arma::mat W = arma::zeros(arma::size(L));
    arma::mat Z = arma::zeros(arma::size(L));

    double curr_loss = arma::datum::inf;

    for (int i = 0; i < maxADMMIterations; ++i) {
        double loss = lossFunction(
            L, L_A, Z - W, target_rho, eta) + lambda * arma::trace(Z);

        // std::cout << loss <<std::endl;
        if (curr_loss - loss < tol_total_loss) {
            break;
        }
        else {
            curr_loss = loss;
        }

        Z = nuclearNormProx(L + W, lambda / eta);
        arma::mat Z_sym = (Z + Z.t()) / 2.0;
        W = L - Z_sym + W;
        // L = L - eps;
        // L += arma::diagmat(arma::sum(-L,1));
        L = centralPath(L, L_A, Z_sym - W, eta, alpha, beta,
            maxGradientDescentIterations, tol, target_rho);
    }

    return L;
}

