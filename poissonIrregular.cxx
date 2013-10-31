static char help [] = "Solves a poisson equation for a regular or irregular grid with Dirichlet Boundary Conditions.";

/*Solves Poisson equation in 3D using finite differences for an irregular domain.
The computational domain (CD) is represented by true/1 values in a 3D binary image
of size xnum X ynum X znum. 
There can be multiple isolated regions of CD.
Uses a standard 7 point stencil discretization of the laplacian.
Two different approaches:
1. Without using PetscSection:
    Here we create matrix for the whole the cube, putting identity rows corresponding
    to the NCD.

2. Using PetscSection:
    Here we create a matrix which would contain rows corresponding only to those points
    that lie in CD.
*/

/*Plan:
1. getRhsAt(i,j,k) --> gives rhs value for any (i,j,k) position within a cube.
2. isCellInDomain(i,j,k) --> returns true if the (i,j,k) cell is to be computed.

Use DMDA to get proper indexing with (i,j,k) matching to the cube.
-------------------------------------------------------------
Create Matrix without using PetscSection:
currentRow = 0
LOOP for all cells in the cube with i,j,k indices:
    IF isCellInDomain(i,j,k):
        currentRow: put a 7 point stencil in the matrix:
            IF NOT(isCellDomain(i+-1,j+-1,k+-1)):
                skip putting anything in the matrix.
    ELSE:
        currentRow: set identity row.
    ++currentRow
-------------------------------------------------------------
Create Rhs Vector without using PetsSection:
index = 0
LOOP for all cells in the cube with i,j,k indices:
        currentElement(index) = rhs(i,j,k)
        IF isCellInDomain(i,j,k):
            #FOR EACH (i+-1,j+-1,k+-1)) OUTSIDE the domain:
                currentElement(index) -= rhs(i+-1,j+-1,k+-1)# --> #A function prolly getBoundaryContributionToRhs(i,j,k)#
    ++index
-------------------------------------------------------------
Create Matrix using PetscSection:
currentRow=0
LOOP for all cells in the cube with i,j,k indices:
    IF isCellInDomain(i,j,k):
        put a row with 7 point stencil in the matrix:
            IF NOT(isCellDomain(i+-1,j+-1,k+-1)):
                skip putting anything in the matrix.
    ++currentRow
-------------------------------------------------------------
Create Rhs Vector using PetsSection:
index = 0
LOOP for all cells in the cube with i,j,k indices:
    IF isCellInDomain(i,j,k):
        currentElement(index) = rhs(i,j,k)
        #FOR EACH (NOT(isCellInDomain(i+-1,j+-1,k+-1))):
            currentElement(index) -= rhs(i+-1,j+-1,k+-1)# --> #A function prolly getBoundaryContributionToRhs(i,j,k)#
    ++index
-------------------------------------------------------------
*/

#include <petsc.h>
#include <petscdm.h>
#include <petscksp.h>

/*User defined structures for the model*/
/*typedef struct {
    PetscScalar       radius;
    PetscScalar       cx,cy,cz;
}Sphere;*/

typedef struct {
    typedef enum {DIRICHLET, NEUMANN} bcType;
    bcType modelBcType;

    int                 mDimension;
    int                 mDof;            /*for velocity poisson equation, mdof = mDimension*/
    int                 *mSize;      /*problem model size {xnum,ynum,znum}.*/
    /*Sphere              mSpheres[2];*/   /*let's have two spheres inside the cube.
    Sphere will later be changed with a binary image.
    This is just for quick test without using image-related libraries.*/
}PoissonModel;

typedef struct {
    PetscScalar         vx,vy;
} Field2d;

typedef struct {
    PetscScalar         vx,vy,vz;
} Field3d;

#undef __FUNCT__
#define __FUNCT__ "modelInitialize"
int modelInitialize(PoissonModel *userModel, const int* size, const int dimension) {
    userModel->modelBcType = PoissonModel::DIRICHLET; /*Currently only Dirichlet boundary supported*/
    userModel->mDimension = dimension;
    userModel->mDof = userModel->mDimension;
    userModel->mSize = (int*) malloc(userModel->mDimension * sizeof(int));
    for( int i = 0; i < userModel->mDimension; ++i)
        userModel->mSize[i] = size[i];
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "modelFinalize"
int modelFinalize(PoissonModel *userModel) {
    free(userModel->mSize);
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "isPosInDomain"
PetscBool isPosInDomain(const PoissonModel *userModel, const int x, const int y, const int z = 0){
    int     pos[3];
    pos[0] = x;     pos[1] = y;     pos[2] = z;
    for(int i = 0; i < userModel->mDimension; ++i)
        if (pos[i] < 0 || pos[i] >= userModel->mSize[i])
            return PETSC_FALSE;

    /*Create a spherical domain:*/
    double      rad1;
    int         c1x, c1y, c1z;
    c1x = userModel->mSize[0]/2;    c1y = userModel->mSize[1]/2;
    if(userModel->mDimension == 2)
        c1z = 0;
    else
        c1z = userModel->mSize[2]/2;
    rad1 = (double)userModel->mSize[0]/3.;
    if (((c1x-x)*(c1x-x) + (c1y-y)*(c1y-y) + (c1z-z)*(c1z-z)) < (rad1*rad1))
        return PETSC_TRUE;
    else
        return PETSC_FALSE;
    /*Later this will read a binary image and return it's value*/
}

#undef __FUNCT__
#define __FUNCT__ "getRhsAt"
double getRhsAt(const PoissonModel *userModel, const int dof, const int x, const int y, const int z = 0) {
    double       dirichletValue = 10;
    /*    if(x < 0) return dirichletValue * 2;*/        /*Bottom horizontal edge*/
    /*Source at the center*/
    if(userModel->mDimension == 2) {
        if(x == userModel->mSize[0]/2 && y == userModel->mSize[1]/2) return 100;
    } else {
        if( x == userModel->mSize[0]/2 && y == userModel->mSize[1]/2 && z == userModel->mSize[2]/2) return 100;
    }

    if(!isPosInDomain(userModel,x,y,z))
        return dirichletValue;
    else
        return 0;
    /*Later this will read an image that contains Rhs values and returns value at requested position*/
}

#undef __FUNCT__
#define __FUNCT__ "computeMatrix2d"
PetscErrorCode computeMatrix2d(KSP ksp, Mat A, Mat B, MatStructure *str, void *context) {
    PoissonModel            *ctx = (PoissonModel*)context;
    PetscErrorCode          ierr;
    DM                      da;
    DMDALocalInfo           info;
    MatStencil              row, col[5];
    PetscScalar             v[5]; /*array to store 5 point stencil for each row*/
    PetscInt                num; /*non-zero position in the current row*/
    PetscScalar             dScale = 1;     /*to scale Dirichlet identity rows if needed*/

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    for(PetscInt dof = 0; dof < 2; ++dof) {
        row.c = dof;
        for(PetscInt k = 0; k < 5; ++k) {
            col[k].c = dof;
        }
        for(PetscInt j = info.ys; j<info.ys+info.ym; ++j) {
            for(PetscInt i = info.xs; i<info.xs+info.xm; ++i) {
                num = 0;
                row.i = i;  row.j = j;
                col[num].i = i; col[num].j = j;
                if (isPosInDomain(ctx,i,j)) {
                    v[num++] = -4;
                    if(isPosInDomain(ctx,i+1,j)) {
                        col[num].i = i+1;   col[num].j = j;
                        v[num++] = 1;
                    }
                    if(isPosInDomain(ctx,i-1,j)) {
                        col[num].i = i-1;   col[num].j = j;
                        v[num++] = 1;
                    }
                    if(isPosInDomain(ctx,i,j+1)) {
                        col[num].i = i;     col[num].j = j+1;
                        v[num++] = 1;
                    }
                    if(isPosInDomain(ctx,i,j-1)) {
                        col[num].i = i;     col[num].j = j-1;
                        v[num++] = 1;
                    }
                } else {
                    v[num++] = dScale;
                }
                ierr = MatSetValuesStencil(A,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*
#undef __FUNCT__
#define __FUNCT__ "computeMatrix2dSection"
PetscErrorCode computeMatrix2dSection(KSP ksp, Mat A, Mat B, MatStructure *str, void *context) {
    PoissonModel            *ctx = (PoissonModel*)context;
    PetscErrorCode          ierr;
    DM                      da;
    DMDALocalInfo           info;
    PetscInt                row, col[5];
    PetscInt                dof;
    PetscScalar             v[5]; //array to store 5 point stencil for each row
    PetscInt                num; //non-zero position in the current row
    PetscScalar             dScale = 1;     //to scale Dirichlet identity rows if needed
    PetscSection            gs;             //Global Section that keeps the grid info and indices
    PetscInt                point;          //Current point of the petscSection

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(da,&gs);CHKERRQ(ierr);
    for(PetscInt j = info.ys; j<info.ys+info.ym; ++j) {
        for(PetscInt i = info.xs; i<info.xs+info.xm; ++i) {
            ierr = DMDAGetCellPoint(da,i,j,0,&point);CHKERRQ(ierr);
            ierr = PetscSectionGetOffset(gs,point,&row);
            ierr = PetscSectionGetDof(gs,point,&dof);
            for(PetscInt cDof = 0; cDof < dof; ++cDof) {
                num = 0;
                row+=cDof;
                col[num] = row;         //(i,j) position
                if (isPosInDomain(ctx,i,j)) {
                    v[num++] = -4;
                    if(isPosInDomain(ctx,i+1,j)) {
                        ierr = DMDAGetCellPoint(da,i+1,j,0,&point);CHKERRQ(ierr);
                        ierr = PetscSectionGetOffset(gs,point,&col[num]);
                        col[num] += cDof;
                        v[num++] = 1;
                    }
                    if(isPosInDomain(ctx,i-1,j)) {
                        ierr = DMDAGetCellPoint(da,i-1,j,0,&point);CHKERRQ(ierr);
                        ierr = PetscSectionGetOffset(gs,point,&col[num]);
                        col[num] += cDof;
                        v[num++] = 1;
                    }
                    if(isPosInDomain(ctx,i,j+1)) {
                        ierr = DMDAGetCellPoint(da,i,j+1,0,&point);CHKERRQ(ierr);
                        ierr = PetscSectionGetOffset(gs,point,&col[num]);
                        col[num] += cDof;
                        v[num++] = 1;
                    }
                    if(isPosInDomain(ctx,i,j-1)) {
                        ierr = DMDAGetCellPoint(da,i,j-1,0,&point);CHKERRQ(ierr);
                        ierr = PetscSectionGetOffset(gs,point,&col[num]);
                        col[num] += cDof;
                        v[num++] = 1;
                    }
                } else {
                    v[num++] = dScale;
                }
                ierr = MatSetValues(A,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
*/

#undef __FUNCT__
#define __FUNCT__ "computeRhs2d"
    PetscErrorCode computeRhs2d(KSP ksp, Vec b, void *context) {
        PetscErrorCode          ierr;
        PoissonModel            *ctx = (PoissonModel*)context;
        DM                      da;
        DMDALocalInfo           info;
        Field2d                 **rhs;

        PetscFunctionBeginUser;
        ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
        ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da,b,&rhs);CHKERRQ(ierr);

        for(PetscInt j = info.ys; j<info.ys+info.ym; ++j) {
            for(PetscInt i = info.xs; i<info.xs+info.xm; ++i) {
                rhs[j][i].vx = getRhsAt(ctx,0,i,j);
                rhs[j][i].vy = getRhsAt(ctx,1,i,j);
                if(isPosInDomain(ctx,i,j)) {
                    if(!isPosInDomain(ctx,i+1,j)) {
                        rhs[j][i].vx -= getRhsAt(ctx,0,i+1,j);
                        rhs[j][i].vy -= getRhsAt(ctx,1,i+1,j);
                    }
                    if(!isPosInDomain(ctx,i-1,j)) {
                        rhs[j][i].vx -= getRhsAt(ctx,0,i-1,j);
                        rhs[j][i].vy -= getRhsAt(ctx,1,i-1,j);
                    }
                    if(!isPosInDomain(ctx,i,j+1)) {
                        rhs[j][i].vx -= getRhsAt(ctx,0,i,j+1);
                        rhs[j][i].vy -= getRhsAt(ctx,1,i,j+1);
                    }
                    if(!isPosInDomain(ctx,i,j-1)) {
                        rhs[j][i].vx -= getRhsAt(ctx,0,i,j-1);
                        rhs[j][i].vy -= getRhsAt(ctx,1,i,j-1);
                    }
                }
            }
        }
        ierr = DMDAVecRestoreArray(da,b,&rhs);CHKERRQ(ierr);


        PetscFunctionReturn(0);

    }

#undef __FUNCT__
#define __FUNCT__ "computeMatrix3d"
    PetscErrorCode computeMatrix3d(KSP ksp, Mat A, Mat B, MatStructure *str, void *context) {
        PoissonModel            *ctx = (PoissonModel*)context;
        PetscErrorCode          ierr;
        DM                      da;
        DMDALocalInfo           info;
        MatStencil              row, col[7];
        PetscScalar             v[7]; /*array to store 7 point stencil for each row*/
        PetscInt                num; /*non-zero position in the current row*/
        PetscScalar             dScale = 1;     /*to scale Dirichlet identity rows if needed*/

        PetscFunctionBeginUser;
        ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
        ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
        for(PetscInt dof = 0; dof < 3; ++dof) {
            row.c = dof;
            for(PetscInt k = 0; k < 7; ++k) {
                col[k].c = dof;
            }
            for(PetscInt k =info.zs; k<info.zs+info.zm; ++k) {
                for(PetscInt j = info.ys; j<info.ys+info.ym; ++j) {
                    for(PetscInt i = info.xs; i<info.xs+info.xm; ++i) {
                        num = 0;
                        row.i = i;  row.j = j;  row.k = k;
                        col[num].i = i; col[num].j = j; col[num].k = k;
                        if (isPosInDomain(ctx,i,j,k)) {
                            v[num++] = -4;
                            if(isPosInDomain(ctx,i+1,j,k)) {
                                col[num].i = i+1;   col[num].j = j; col[num].k = k;
                                v[num++] = 1;
                            }
                            if(isPosInDomain(ctx,i-1,j,k)) {
                                col[num].i = i-1;   col[num].j = j; col[num].k = k;
                                v[num++] = 1;
                            }
                            if(isPosInDomain(ctx,i,j+1,k)) {
                                col[num].i = i;     col[num].j = j+1;   col[num].k = k;
                                v[num++] = 1;
                            }
                            if(isPosInDomain(ctx,i,j-1,k)) {
                                col[num].i = i;     col[num].j = j-1;   col[num].k = k;
                                v[num++] = 1;
                            }
                            if(isPosInDomain(ctx,i,j,k+1)) {
                                col[num].i = i;     col[num].j = j;     col[num].k = k+1;
                                v[num++] = 1;
                            }
                            if(isPosInDomain(ctx,i,j,k-1)) {
                                col[num].i = i;     col[num].j = j;     col[num].k = k-1;
                                v[num++] = 1;
                            }
                        } else {
                            v[num++] = dScale;
                        }
                        ierr = MatSetValuesStencil(A,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
                    }
                }
            }
        }

        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        PetscFunctionReturn(0);
    }

#undef __FUNCT__
#define __FUNCT__ "computeRhs3d"
    PetscErrorCode computeRhs3d(KSP ksp, Vec b, void *context) {
        PetscErrorCode          ierr;
        PoissonModel            *ctx = (PoissonModel*)context;
        DM                      da;
        DMDALocalInfo           info;
        Field3d                 ***rhs;

        PetscFunctionBeginUser;
        ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
        ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da,b,&rhs);CHKERRQ(ierr);

        for(PetscInt k = info.zs; k<info.zs+info.zm; ++k) {
            for(PetscInt j = info.ys; j<info.ys+info.ym; ++j) {
                for(PetscInt i = info.xs; i<info.xs+info.xm; ++i) {
                    rhs[k][j][i].vx = getRhsAt(ctx,0,i,j,k);
                    rhs[k][j][i].vy = getRhsAt(ctx,1,i,j,k);
                    rhs[k][j][i].vz = getRhsAt(ctx,2,i,j,k);
                    if(isPosInDomain(ctx,i,j,k)) {
                        if(!isPosInDomain(ctx,i+1,j,k)) {
                            rhs[k][j][i].vx -= getRhsAt(ctx,0,i+1,j,k);
                            rhs[k][j][i].vy -= getRhsAt(ctx,1,i+1,j,k);
                            rhs[k][j][i].vz -= getRhsAt(ctx,2,i+1,j,k);
                        }
                        if(!isPosInDomain(ctx,i-1,j,k)) {
                            rhs[k][j][i].vx -= getRhsAt(ctx,0,i-1,j,k);
                            rhs[k][j][i].vy -= getRhsAt(ctx,1,i-1,j,k);
                            rhs[k][j][i].vz -= getRhsAt(ctx,2,i-1,j,k);
                        }
                        if(!isPosInDomain(ctx,i,j+1,k)) {
                            rhs[k][j][i].vx -= getRhsAt(ctx,0,i,j+1,k);
                            rhs[k][j][i].vy -= getRhsAt(ctx,1,i,j+1,k);
                            rhs[k][j][i].vz -= getRhsAt(ctx,2,i,j+1,k);
                        }
                        if(!isPosInDomain(ctx,i,j-1,k)) {
                            rhs[k][j][i].vx -= getRhsAt(ctx,0,i,j-1,k);
                            rhs[k][j][i].vy -= getRhsAt(ctx,1,i,j-1,k);
                            rhs[k][j][i].vz -= getRhsAt(ctx,2,i,j-1,k);
                        }
                        if(!isPosInDomain(ctx,i,j,k+1)) {
                            rhs[k][j][i].vx -= getRhsAt(ctx,0,i,j,k+1);
                            rhs[k][j][i].vy -= getRhsAt(ctx,1,i,j,k+1);
                            rhs[k][j][i].vz -= getRhsAt(ctx,2,i,j,k+1);
                        }
                        if(!isPosInDomain(ctx,i,j,k-1)) {
                            rhs[k][j][i].vx -= getRhsAt(ctx,0,i,j,k-1);
                            rhs[k][j][i].vy -= getRhsAt(ctx,1,i,j,k-1);
                            rhs[k][j][i].vz -= getRhsAt(ctx,2,i,j,k-1);
                        }
                    }
                }
            }
        }
        ierr = DMDAVecRestoreArray(da,b,&rhs);CHKERRQ(ierr);


        PetscFunctionReturn(0);

    }

#undef __FUNCT__
#define __FUNCT__ "main"
    int main(int argc, char** args) {
        PetscInt      dim;
        PetscInt            *size;
        PoissonModel        testPoisson;
        PetscErrorCode      ierr;
        PetscBool           optionFlag;
        DM                  dm;
        KSP                 ksp;
        Mat                 A;
        Vec                 x,b,res;
        MPI_Comm            comm = PETSC_COMM_WORLD;

        PetscInitialize(&argc,&args,(char*)0,help);
        ierr = PetscOptionsGetInt(PETSC_NULL,"-dim",&dim,&optionFlag);CHKERRQ(ierr);
        if(!optionFlag) dim = 2;        /*default dimension 2*/
        if(dim !=2 && dim !=3) SETERRQ(comm,PETSC_ERR_SUP,"currently only 2D or 3D model implemented");
        size = (PetscInt*) malloc(dim*sizeof(PetscInt));
        ierr = PetscOptionsGetInt(PETSC_NULL,"-x_size",&size[0],&optionFlag);CHKERRQ(ierr);
        if(!optionFlag) size[0] = 10;   /*default size*/
        ierr = PetscOptionsGetInt(PETSC_NULL,"-y_size",&size[1],&optionFlag);CHKERRQ(ierr);
        if(!optionFlag) size[1] = 10;   /*defalut size*/
        if(dim == 3) {
            ierr = PetscOptionsGetInt(PETSC_NULL,"-z_size",&size[2],&optionFlag);CHKERRQ(ierr);
            if(!optionFlag) size[2] = 10; /*default size*/
        }
        modelInitialize(&testPoisson,size,dim);
        if (testPoisson.mDimension == 2) {
            ierr = DMDACreate2d(comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,
                                size[0],size[1],PETSC_DECIDE,PETSC_DECIDE,testPoisson.mDof,1,NULL,NULL,&dm);CHKERRQ(ierr);
            ierr = DMDASetFieldName(dm,0,"vx");CHKERRQ(ierr);
            ierr = DMDASetFieldName(dm,1,"vy");CHKERRQ(ierr);
        } else {
            ierr = DMDACreate3d(comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,
                                size[0],size[1],size[2],PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,testPoisson.mDof,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
            ierr = DMDASetFieldName(dm,0,"vx");CHKERRQ(ierr);
            ierr = DMDASetFieldName(dm,1,"vy");CHKERRQ(ierr);
            ierr = DMDASetFieldName(dm,2,"vz");CHKERRQ(ierr);
        }

        /*START PetscSection stuff*/
        /*PetscSection            s;
        PetscInt                nC;
        DMDALocalInfo           info;
        ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
        ierr = PetscSectionCreate(comm, &s);CHKERRQ(ierr);
        ierr = DMDAGetNumCells(dm, NULL, NULL, NULL, &nC);CHKERRQ(ierr);
        ierr = PetscSectionSetChart(s, 0, nC);CHKERRQ(ierr);
        for (PetscInt j = info.ys; j < info.ys+info.ym; ++j) {
            for (PetscInt i = info.xs; i < info.xs+info.xm; ++i) {
                PetscInt point;
                if(isPosInDomain(&testPoisson,i,j,0)) {
                    ierr = DMDAGetCellPoint(dm, i, j, 0, &point);CHKERRQ(ierr);
                    ierr = PetscSectionSetDof(s, point, testPoisson.mDof); // No. of dofs associated with the point.
                }

            }
        }
        ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
        ierr = DMSetDefaultSection(dm, s);CHKERRQ(ierr);
        ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);*/

        /*END PetscSection stuff*/

        //Solve the model:
        ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
        ierr = KSPSetDM(ksp,dm);CHKERRQ(ierr);
        if (testPoisson.mDimension == 2) {
            ierr = KSPSetComputeOperators(ksp,computeMatrix2d,(void*)&testPoisson);CHKERRQ(ierr);
//            ierr = KSPSetComputeOperators(ksp,computeMatrix2dSection,(void*)&testPoisson);CHKERRQ(ierr);
            ierr = KSPSetComputeRHS(ksp,computeRhs2d,(void*)&testPoisson);CHKERRQ(ierr);
        } else if (testPoisson.mDimension == 3) {
            ierr = KSPSetComputeOperators(ksp,computeMatrix3d,(void*)&testPoisson);CHKERRQ(ierr);
            ierr = KSPSetComputeRHS(ksp,computeRhs3d,(void*)&testPoisson);CHKERRQ(ierr);
        }
        ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
        ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
        ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
        ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);
        ierr = KSPGetOperators(ksp,&A,NULL,NULL);CHKERRQ(ierr);

        /*compute Residual*/
        ierr = VecDuplicate(x,&res);CHKERRQ(ierr);
        ierr = VecSet(res,0);CHKERRQ(ierr);
        ierr = VecAXPY(res,-1.0,b);CHKERRQ(ierr);
        ierr = MatMultAdd(A,x,res,res);CHKERRQ(ierr);

        /*Write solution, rhs and residuals to file*/
        PetscViewer         viewer;
        /*ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,NULL,"solution vector",PETSC_DECIDE,PETSC_DECIDE,
                               PETSC_DRAW_HALF_SIZE,PETSC_DRAW_HALF_SIZE,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawSetPause(viewer,-1);CHKERRQ(ierr);*/
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"sol",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)x,"x");CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)b,"b");CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)res,"res");CHKERRQ(ierr);
        ierr = VecView(x,viewer);CHKERRQ(ierr);
        ierr = VecView(b,viewer);CHKERRQ(ierr);
        ierr = VecView(res,viewer);CHKERRQ(ierr);

        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        ierr = VecDestroy(&res);CHKERRQ(ierr);

        /*PetscViewer viewer2;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"sysFile",FILE_MODE_WRITE,&viewer2);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer2,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)A,"A");CHKERRQ(ierr);
    ierr = MatView(A,viewer2);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer2);CHKERRQ(ierr);
*/
        modelFinalize(&testPoisson);
        free(size);

        PetscFinalize();
        return 0;
    }
