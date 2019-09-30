#ifndef OSQPMODULEMETHODS_H
#define OSQPMODULEMETHODS_H

/***********************************************************************
 * OSQP methods independently from any object                          *
 ***********************************************************************/

static PyObject * OSQP_module_solve(OSQP *self, PyObject *args, PyObject *kwargs) {
  c_int n, m;  // Problem dimensions
  c_int exitflag_setup, exitflag_solve, status_val;

  // Variables for setup
  PyOSQPData   *pydata;
  OSQPData     *data;
  OSQPSettings *settings;
  OSQPSolver   *solver;  // Pointer to C solver structure
  PyArrayObject *Px, *Pi, *Pp, *q, *Ax, *Ai, *Ap, *l, *u;


  // Variables for solve
  // Create status object
  PyObject *status;
  // Create obj_val object
  PyObject *obj_val;
  // Create solution objects
  PyObject *x, *y, *prim_inf_cert, *dual_inf_cert;
  // Define info related variables
  static char *argparse_string_info;
  PyObject *info_list;
  PyObject *info;

  // Results
  PyObject *results_list;
  PyObject *results;

  npy_intp nd[1];
  npy_intp md[1];

  printf("000\n");

  static char *kwlist[] = {"dims",                     // nvars and ncons
			   "Px", "Pi", "Pp", "q",      // Cost function
			   "Ax", "Ai", "Ap", "l", "u", // Constraints
			   "scaling",
			   "adaptive_rho", "adaptive_rho_interval",
			   "adaptive_rho_tolerance", "adaptive_rho_fraction",
			   "rho", "sigma", "max_iter", "eps_abs", "eps_rel",
			   "eps_prim_inf", "eps_dual_inf", "alpha", "delta",
			   "linsys_solver", "polish",
			   "polish_refine_iter", "verbose",
			   "scaled_termination",
			   "check_termination", "warm_start",
			   "time_limit", NULL};        // Settings

#ifdef DLONG

  // NB: linsys_solver is enum type which is stored as int (regardless on how c_int is defined).
#ifdef DFLOAT
  static char * argparse_string_setup = "(LL)O!O!O!O!O!O!O!O!O!|LLLffffLffffffiLLLLLLf";
#else
  static char * argparse_string_setup = "(LL)O!O!O!O!O!O!O!O!O!|LLLddddLddddddiLLLLLLd";
#endif

#else

#ifdef DFLOAT
  static char * argparse_string_setup = "(ii)O!O!O!O!O!O!O!O!O!|iiiffffiffffffiiiiiiif";
#else
  static char * argparse_string_setup = "(ii)O!O!O!O!O!O!O!O!O!|iiiddddiddddddiiiiiiid";
#endif

#endif

  // Initialize settings
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);

  if( !PyArg_ParseTupleAndKeywords(args, kwargs, argparse_string_setup, kwlist,
				   &n, &m,
				   &PyArray_Type, &Px,
				   &PyArray_Type, &Pi,
				   &PyArray_Type, &Pp,
				   &PyArray_Type, &q,
				   &PyArray_Type, &Ax,
				   &PyArray_Type, &Ai,
				   &PyArray_Type, &Ap,
				   &PyArray_Type, &l,
				   &PyArray_Type, &u,
				   &settings->scaling,
				   &settings->adaptive_rho,
				   &settings->adaptive_rho_interval,
				   &settings->adaptive_rho_tolerance,
				   &settings->adaptive_rho_fraction,
				   &settings->rho,
				   &settings->sigma,
				   &settings->max_iter,
				   &settings->eps_abs,
				   &settings->eps_rel,
				   &settings->eps_prim_inf,
				   &settings->eps_dual_inf,
				   &settings->alpha,
				   &settings->delta,
				   &settings->linsys_solver,
				   &settings->polish,
				   &settings->polish_refine_iter,
				   &settings->verbose,
				   &settings->scaled_termination,
				   &settings->check_termination,
				   &settings->warm_start,
				   &settings->time_limit)) {
    return (PyObject *) NULL;
  }

  printf("111\n");

  // Create Data from parsed vectors
  pydata = create_pydata(n, m, Px, Pi, Pp, q, Ax, Ai, Ap, l, u);
  data = create_data(pydata);

  // Perform setup and solve
  // release the GIL
  Py_BEGIN_ALLOW_THREADS;

  printf("222\n");

  // Create Solver object
  exitflag_setup = osqp_setup(&solver, data->P, data->q, data->A,
                              data->l, data->u, data->m, data->n, settings);

  printf("333\n");

  exitflag_solve = osqp_solve(solver);
  // reacquire the GIL
  Py_END_ALLOW_THREADS;

  printf("444\n");
  
  // Cleanup data and settings
  free_data(data, pydata);
  c_free(settings);

  // Check successful setup and solve
  if (exitflag_setup){ // Solver allocation error
    PyErr_SetString(PyExc_ValueError, "Solver allocation error!");
    return (PyObject *) NULL;
  }

  if(exitflag_solve){
      PyErr_SetString(PyExc_ValueError, "OSQP solve error!");
      return (PyObject *) NULL;
  }

  // Temporary solution
  osqp_get_dimensions(self->solver, &m, &n);
  nd[0] = (npy_intp)n;  // Dimensions in R^n
  md[0] = (npy_intp)m;  // Dimensions in R^m

  // If problem is not primal or dual infeasible store it
  status_val = self->solver->info->status_val;
  if ((status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (status_val != OSQP_DUAL_INFEASIBLE) &&
      (status_val != OSQP_DUAL_INFEASIBLE_INACCURATE)){

    // Primal and dual solutions
    x = (PyObject *)PyArrayFromCArray(solver->solution->x, nd);
    y = (PyObject *)PyArrayFromCArray(solver->solution->y, md);

    // Infeasibility certificates -> None values
    prim_inf_cert = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
    dual_inf_cert = PyArray_EMPTY(1, md, NPY_OBJECT, 0);

  } else if (status_val == OSQP_PRIMAL_INFEASIBLE ||
	           status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE) {
    // primal infeasible

    // Primal and dual solution arrays -> None values
    x = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
    y = PyArray_EMPTY(1, md, NPY_OBJECT, 0);

    // Primal infeasibility certificate
    prim_inf_cert = (PyObject *)PyArrayFromCArray(solver->solution->prim_inf_cert, md);

    // Dual infeasibility certificate -> None values
    dual_inf_cert = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);

    // Set objective value to infinity
    solver->info->obj_val = NPY_INFINITY;

  } else {
    // dual infeasible

    // Primal and dual solution arrays -> None values
    x = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
    y = PyArray_EMPTY(1, md, NPY_OBJECT, 0);

    // Primal infeasibility certificate -> None values
    prim_inf_cert = PyArray_EMPTY(1, md, NPY_OBJECT, 0);

    // Dual infeasibility certificate
    dual_inf_cert = (PyObject *)PyArrayFromCArray(solver->solution->dual_inf_cert, nd);

    // Set objective value to -infinity
    solver->info->obj_val = -NPY_INFINITY;
  }

  /*  CREATE INFO OBJECT */
  // Store status string
  status = PyUnicode_FromString(solver->info->status);

  // Store obj_val
  if (status_val == OSQP_NON_CVX) {	// non convex
    obj_val = PyFloat_FromDouble(Py_NAN);
  } else {
    obj_val = PyFloat_FromDouble(solver->info->obj_val);
  }

#ifdef PROFILING

#ifdef DLONG

#ifdef DFLOAT
  argparse_string_info = "LOLLfLOfffffff";
#else
  argparse_string_info = "LOLLdLOddddddd";
#endif

#else

#ifdef DFLOAT
  argparse_string_info = "iOiifiOfffffff";
#else
  argparse_string_info = "iOiidiOddddddd";
#endif

#endif /* DLONG */

  info_list = Py_BuildValue(argparse_string_info,
			    solver->info->iter,
			    status,
			    solver->info->status_val,
          solver->info->rho_updates,
			    solver->info->rho_estimate,
			    solver->info->status_polish,
			    obj_val,
			    solver->info->pri_res,
			    solver->info->dua_res,
			    solver->info->setup_time,
			    solver->info->solve_time,
			    solver->info->update_time,
			    solver->info->polish_time,
			    solver->info->run_time
			    );
#else /* PROFILING */

#ifdef DLONG

#ifdef DFLOAT
  argparse_string = "LOLLfLOff";
#else
  argparse_string = "LOLLdLOdd";
#endif

#else

#ifdef DFLOAT
  argparse_string = "iOiifiOff";
#else
  argparse_string = "iOiidiOdd";
#endif

#endif /* DLONG */

  info_list = Py_BuildValue(argparse_string_info,
			    solver->info->iter,
			    status,
			    solver->info->status_val,
          solver->info->rho_updates,
			    solver->info->rho_estimate,
			    solver->info->status_polish,
			    obj_val,
			    solver->info->pri_res,
			    solver->info->dua_res
			    );
#endif /* PROFILING */

  printf("555\n");

  info = PyObject_CallObject((PyObject *) &OSQP_info_Type, info_list);

  /* Release the info argument list. */
  Py_DECREF(info_list);

  /*  CREATE RESULTS OBJECT */
  results_list = Py_BuildValue("OOOOO", x, y, prim_inf_cert, dual_inf_cert, info);

  /* Call the class object. */
  results = PyObject_CallObject((PyObject *) &OSQP_results_Type, results_list);

  // Delete results list
  Py_DECREF(results_list);

  // Cleanup solver
  if (osqp_cleanup(solver)) {
    PyErr_SetString(PyExc_ValueError, "Solver deallocation error!");
    return (PyObject *) NULL;
  }

  printf("555\n");

  // Return results    
  return results;


}


static PyObject *OSQP_constant(OSQP *self, PyObject *args) {

    char * constant_name;  // String less than 32 chars

    // Parse arguments
    if( !PyArg_ParseTuple(args, "s", &(constant_name))) {
        return (PyObject *) NULL;
    }


    if(!strcmp(constant_name, "OSQP_INFTY")){
#ifdef DFLOAT
        return Py_BuildValue("f", OSQP_INFTY);
#else
        return Py_BuildValue("d", OSQP_INFTY);
#endif
    }

    if(!strcmp(constant_name, "OSQP_NAN")){
#ifdef DFLOAT
        return Py_BuildValue("f", Py_NAN);
#else
        return Py_BuildValue("d", Py_NAN);
#endif
    }

    if(!strcmp(constant_name, "OSQP_SOLVED")){
        return Py_BuildValue("i", OSQP_SOLVED);
    }

    if(!strcmp(constant_name, "OSQP_SOLVED_INACCURATE")){
        return Py_BuildValue("i", OSQP_SOLVED_INACCURATE);
    }

    if(!strcmp(constant_name, "OSQP_UNSOLVED")){
        return Py_BuildValue("i", OSQP_UNSOLVED);
    }

    if(!strcmp(constant_name, "OSQP_PRIMAL_INFEASIBLE")){
        return Py_BuildValue("i", OSQP_PRIMAL_INFEASIBLE);
    }

	if(!strcmp(constant_name, "OSQP_PRIMAL_INFEASIBLE_INACCURATE")){
		return Py_BuildValue("i", OSQP_PRIMAL_INFEASIBLE_INACCURATE);
	}

    if(!strcmp(constant_name, "OSQP_DUAL_INFEASIBLE")){
        return Py_BuildValue("i", OSQP_DUAL_INFEASIBLE);
    }

	if(!strcmp(constant_name, "OSQP_DUAL_INFEASIBLE_INACCURATE")){
		return Py_BuildValue("i", OSQP_DUAL_INFEASIBLE_INACCURATE);
	}

    if(!strcmp(constant_name, "OSQP_MAX_ITER_REACHED")){
        return Py_BuildValue("i", OSQP_MAX_ITER_REACHED);
    }

    if(!strcmp(constant_name, "OSQP_NON_CVX")){
        return Py_BuildValue("i", OSQP_NON_CVX);
    }

    if(!strcmp(constant_name, "OSQP_TIME_LIMIT_REACHED")){
        return Py_BuildValue("i", OSQP_TIME_LIMIT_REACHED);
    }

	// Linear system solvers
	if(!strcmp(constant_name, "QDLDL_SOLVER")){
		return Py_BuildValue("i", QDLDL_SOLVER);
	}

	if(!strcmp(constant_name, "MKL_PARDISO_SOLVER")){
		return Py_BuildValue("i", MKL_PARDISO_SOLVER);
	}

    // If reached here error
    PyErr_SetString(PyExc_ValueError, "Constant not recognized");
    return (PyObject *) NULL;
}




static PyMethodDef OSQP_module_methods[] = {
					    {"solve", (PyCFunction)OSQP_module_solve,METH_VARARGS|METH_KEYWORDS, PyDoc_STR("Setup solve and cleanup OSQP problem. This function releases GIL.")},
					    {"constant", (PyCFunction)OSQP_constant, METH_VARARGS, PyDoc_STR("Return internal OSQP constant")},
					    {NULL, NULL}		/* sentinel */
};

  
#endif
